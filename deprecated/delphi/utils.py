import numpy as np
import pandas as pd
import torch
import re
import os


def get_p2i(data):
    patient_ids = data[:, 0].astype(int)
    _, idx_start, counts = np.unique(patient_ids, return_index=True, return_counts=True)
    return np.stack([idx_start, counts], axis=1)


def get_batch(ix, data, p2i, select='left', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False, 
              no_event_token_rate=5, cut_batch=False, return_subject_ids=False):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'center', 'right', 'smart_random', 'smart_right'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """

    MASKING_TOKEN, MASKING_AGE = -1, -10000
    LIFESTYLE_MIN_INDEX, LIFESTYLE_MAX_INDEX = 3, 11

    # Define the columns of the data array    
    SUBJECT_ID_COLUMN = 0
    AGE_COLUMN = 1
    TOKEN_COLUMN = 2

    subject_start_and_count = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    if return_subject_ids:
        subject_ids = torch.tensor(np.array([data[int(subject_index[0]), SUBJECT_ID_COLUMN] for subject_index in subject_start_and_count]))        
            
    if isinstance(ix, list) or isinstance(ix, range):
        ix = torch.tensor(np.array(ix))
    if isinstance(ix, np.ndarray):
        ix = torch.from_numpy(ix)
    if isinstance(ix, torch.Tensor):
        pass

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = subject_start_and_count[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(subject_start_and_count[:, 0] + subject_start_and_count[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = subject_start_and_count[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(subject_start_and_count[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError(f"Selection method {select} not implemented.")
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, SUBJECT_ID_COLUMN][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.cpu().numpy()][:, SUBJECT_ID_COLUMN], SUBJECT_ID_COLUMN][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, TOKEN_COLUMN][batch_idx].astype(np.int64))
    ages   = torch.from_numpy(data[:, AGE_COLUMN][batch_idx].astype(np.float32))

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= LIFESTYLE_MIN_INDEX) * (tokens <= LIFESTYLE_MAX_INDEX)
        n_lifestyles_tokens = lifestyle_idx.sum()
        if n_lifestyles_tokens:
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (n_lifestyles_tokens,), generator=gen).float()

    tokens = tokens.masked_fill(~mask, MASKING_TOKEN)
    ages   = ages.masked_fill(~mask, MASKING_AGE)

    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    tokens = tokens.masked_fill(ages > m, MASKING_TOKEN)
    ages = ages.masked_fill(ages > m, MASKING_AGE)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # cut to maintain the block size
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # shift by one to generate targets
    x, y = tokens[:, :-1], tokens[:, 1:]
    a, b = ages[:, :-1]  , ages[:, 1:]

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, MASKING_AGE)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)

    if return_subject_ids:
        return x, a, y, b, subject_ids
    else:           
        return x, a, y, b


def get_person(idx):
    x, y, _, time = get_batch([idx], val, val_p2i,  
              select='left', block_size=64, 
              device=device, padding='random', 
              cut_batch=True)
    
    x, y = x[y > -1], y[y > -1]
    person = []
    for token_id, date in zip(x, y):
        person.append((id_to_token[token_id.item()], date.item()))
    return person, y, time[0][-1]


# ——————————————————————————————————————————————————————————————————————————————————————————

from torch.utils.data import Dataset

class DelphiData(Dataset):

    def __init__(self, data_dir, val_fold, delphi_labels, labels, device=None):

        self.data_dir = data_dir
        self.delphi_labels = pd.read_csv(delphi_labels)
        self.labels = pd.read_csv(labels, header=None, sep="\t")
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.val_fold = val_fold
        
        self.get_p2i()
        self.get_id_to_token()


    def tokens_to_ids(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[int(id_)] for id_ in ids]

    def split_person(self, p):
        tokens = [i[0] for i in p]
        ages = [i[1] for i in p]
        return tokens, ages


    @classmethod
    def from_runid(cls, runid):
        
        # data_dir, val_fold, delphi_labels, labels,
        # cls()
        return None


    # ——————————————————————————————————————————————————————————————————————————————————————————
    def get_p2i(self):
        load_data_from_bin = lambda datadir, file: np.fromfile(os.path.join(datadir, file), dtype=np.uint32).reshape(-1, 3)
        
        train_folds = []
        for fold_i in [1, 2, 3, 4, 5]:
            if fold_i == self.val_fold:
                continue
            train_fold_filename = f'fold{fold_i}.bin'
            train_datafold = load_data_from_bin(self.data_dir, train_fold_filename)
            train_folds.append(train_datafold)
        self.train_data = np.concatenate(train_folds)
        
        self.val_filename = f'fold{self.val_fold}.bin'
        self.val_data = load_data_from_bin(self.data_dir, self.val_filename)
        
        self.train_p2i = get_p2i(self.train_data)
        self.val_p2i = get_p2i(self.val_data)
                
    def get_id_to_token(self):
        self.id_to_token = self.labels.to_dict()[0]
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    # ——————————————————————————————————————————————————————————————————————————————————————————
    def get_person(self, idx, data_type="val"):
        """
        Returns the person data for the given index.
        data_type: "val", "train", or "all"
        """
        if data_type == "val":
            data = self.val_data
            p2i = self.val_p2i
            indices = [idx] if isinstance(idx, int) else idx
            x, y, _, time = get_batch(indices, data, p2i,
                                      select='left', block_size=64,
                                      device=self.device, padding='random',
                                      cut_batch=True)
            x, y = x[y > -1], y[y > -1]
            person = []
            for token_id, date in zip(x, y):
                person.append((self.id_to_token[token_id.item()], date.item()))
            return person, y, time[0][-1]
        elif data_type == "train":
            data = self.train_data
            p2i = self.train_p2i
            indices = [idx] if isinstance(idx, int) else idx
            x, y, _, time = get_batch(indices, data, p2i,
                                      select='left', block_size=64,
                                      device=self.device, padding='random',
                                      cut_batch=True)
            x, y = x[y > -1], y[y > -1]
            person = []
            for token_id, date in zip(x, y):
                person.append((self.id_to_token[token_id.item()], date.item()))
            return person, y, time[0][-1]
        elif data_type == "all":
            # Concatenate train and val data
            all_data = np.concatenate([self.train_data, self.val_data])
            all_p2i = get_p2i(all_data)
            indices = [idx] if isinstance(idx, int) else idx
            x, y, _, time = get_batch(indices, all_data, all_p2i,
                                      select='left', block_size=64,
                                      device=self.device, padding='random',
                                      cut_batch=True)
            x, y = x[y > -1], y[y > -1]
            person = []
            for token_id, date in zip(x, y):
                person.append((self.id_to_token[token_id.item()], date.item()))
            return person, y, time[0][-1]
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'val', 'train', or 'all'.")

    # ——————————————————————————————————————————————————————————————————————————————————————————


def get_best_ckpt_from_mlflow(runid):
    
    import mlflow
    from pathlib import Path
    runinfo = mlflow.get_run(run_id=runid)
    
    artifact_dir = Path(
        re.sub(r'^.*(?=mlruns)', 
        os.path.dirname(mlflow.get_tracking_uri()) + "/", 
        runinfo.info.artifact_uri))

    ckpt_dir = artifact_dir / 'checkpoints'
    best_ckpt = [x for x in os.listdir(ckpt_dir) if 'best' in x][0]
    best_ckpt = os.path.join(ckpt_dir, best_ckpt)
    
    return best_ckpt


import mlflow
import re
import os
from typing import List, Dict, Tuple, Optional, Union

DATA_TYPE_CONFIGS = {

    'real-hla-4digits-5folds': {
        'labels': './data/ukb_real_5_folds_4digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla4d.csv',
        'data_root': './data/ukb_real_5_folds_4digit',
    },
    'real-nohla-5folds': {
        'labels': './data/transforms/ukb_real_5_folds_nohla/labels.csv',
        'delphi_labels': './data/delphi_labels_chapters_colours_icd.csv',
        'data_root': './data/transforms/ukb_real_5_folds_nohla',
    },
    'real-hla-2digits-5folds': {
        'labels': './data/ukb_real_5_folds_2digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla2d.csv',
        'data_root': './data/ukb_real_5_folds_2digit',
    },
}


def get_run_from_fold(experiment_id=None, val_fold=None):
    """ 
    Retrieves val_fold and the checkpoint path from MLflow metadata using experiment_id.
    Ensures the experiment contains all 5 folds.
    Returns (val_fold, run_id, ckpt_path).
    """

    experiment_id = str(experiment_id)

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000
    )
    
    if not runs:
        raise ValueError(f"No successful runs found in experiment {experiment_id}")

    val_folds = []
    fold_to_run = {}
    for run in runs:
        val_fold_run = None
        if 'fold' in run.data.params:
            val_fold_run = int(run.data.params['fold'])
        elif 'fold' in run.data.tags:
            val_fold_run = int(run.data.tags['fold'])
        if val_fold_run is not None:
            if val_fold_run not in fold_to_run:
                fold_to_run[val_fold_run] = run 
                val_folds.append(val_fold_run)

    if sorted(val_folds) != [1, 2, 3, 4, 5]: 
        raise ValueError(f"Experiment {experiment_id} does not contain all 5 folds! Found folds: {sorted(val_folds)}")

    requested_fold = val_fold

    if requested_fold not in fold_to_run:
        raise ValueError(f"Requested fold {requested_fold} not found in experiment {experiment_id}")

    run = fold_to_run[requested_fold]

    return run.info.run_id


def load_fold_ids(fold_dir: str, num_folds: int = 10) -> List[List[str]]:
    """
    Load subject IDs from predefined fold files.
    Each file must be named `fold_i_of_num.txt` (0-indexed).
    """
    folds = []
    for i in range(1, num_folds+1):
        fname = os.path.join(fold_dir, f"subset{i}of{num_folds}.csv")
        with open(fname) as f:
            ids = [line.strip() for line in f if line.strip()]
            folds.append(ids)
    return folds


def generate_splits(
    folds: List[List[str]],
    n_train_folds: int,
    n_val_folds: int,
    n_test_folds: int,
    val_as_last: bool = True,
) -> List[Dict[str, List[str]]]:
    """
    Generate splits (train/valid/test) given predefined folds.

    Parameters
    ----------
    folds : list of list
        List of folds, each containing subject IDs.
    n_train_folds : int
        Number of folds to use for training.
    n_val_folds : int
        Number of folds to use for validation.
    n_test_folds : int
        Number of folds to use for testing.
    val_as_last : bool
        If True, take validation folds as the last `n_val_folds` among the remaining.
        If False, take the first `n_val_folds`.

    Returns
    -------
    splits : list of dict
        Each dict has keys "train", "valid", "test".
    """
    num_folds = len(folds)
    window_size = n_train_folds + n_val_folds + n_test_folds
    if window_size > num_folds:
        raise ValueError("Not enough folds for requested split sizes.")

    splits = []
    for start in range(0, num_folds, n_test_folds):
        test_idx = list(range(start, start + n_test_folds))
        remaining = [i for i in range(num_folds) if i not in test_idx]

        if val_as_last:
            val_idx = remaining[-n_val_folds:]
            train_idx = remaining[:-n_val_folds]
        else:
            val_idx = remaining[:n_val_folds]
            train_idx = remaining[n_val_folds:n_val_folds + n_train_folds]

        split = {
            "train": sum([folds[i] for i in train_idx], []),
            "valid": sum([folds[i] for i in val_idx], []),
            "test": sum([folds[i] for i in test_idx], []),
        }
        splits.append(split)

    return splits
