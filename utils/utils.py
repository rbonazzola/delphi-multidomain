import numpy as np
import pandas as pd
import torch
import re
import os
import ast 

PADDING_TOKEN = 0
NO_EVENT_TOKEN_ID = 1
MASKING_TOKEN, MASKING_AGE = -1, -10000
LIFESTYLE_MIN_INDEX, LIFESTYLE_MAX_INDEX = 3, 11

is_lifestyle_token = lambda tokens: (tokens >= LIFESTYLE_MIN_INDEX) * (tokens <= LIFESTYLE_MAX_INDEX)

def get_p2i(data):
    patient_ids = data[:, 0].astype(int)
    _, idx_start, counts = np.unique(patient_ids, return_index=True, return_counts=True)
    return np.stack([idx_start, counts], axis=1)


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
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

    # Define the columns of the data array    
    SUBJECT_ID_COLUMN, AGE_COLUMN, TOKEN_COLUMN= 0, 1, 2

    subject_start_and_count = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    if return_subject_ids:
        subject_ids = torch.tensor(np.array([data[int(subject_index[0]), SUBJECT_ID_COLUMN] for subject_index in subject_start_and_count]))        
    ix = torch.tensor(np.array(ix))
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
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, SUBJECT_ID_COLUMN][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(
        data[p2i[ix.numpy()][:, SUBJECT_ID_COLUMN], SUBJECT_ID_COLUMN][:, None].astype(np.int64)
    ).to(mask.dtype)

    tokens = torch.from_numpy(data[:, TOKEN_COLUMN][batch_idx].astype(np.int64))
    ages   = torch.from_numpy(data[:, AGE_COLUMN][batch_idx].astype(np.float32))

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = is_lifestyle_token(tokens) # (tokens >= LIFESTYLE_MIN_INDEX) * (tokens <= LIFESTYLE_MAX_INDEX)
        n_lifestyles_tokens = lifestyle_idx.sum()
        if n_lifestyles_tokens:
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (n_lifestyles_tokens,), generator=gen).float()

    tokens = tokens.masked_fill(~mask, MASKING_TOKEN)
    ages   = ages.masked_fill(~mask, MASKING_AGE)

    # insert a "no event" token every "no_event_token_rate" years on average
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

    # invalid_high = (tokens >= vocab_size)
    # invalid_low = (tokens < 0)
    # 
    # if invalid_high.any():
    #     idx = torch.nonzero(invalid_high)
    #     print(f"🛑 Token(s) con índice demasiado alto detectados:")
    #     for i in idx:
    #         print(f" - Posición {tuple(i.tolist())}, valor: {tokens[tuple(i.tolist())].item()}, vocab_size: {vocab_size}")
    #     raise ValueError("Se encontraron índices fuera del rango superior del vocabulario.")
    # 
    # if invalid_low.any():
    #     idx = torch.nonzero(invalid_low)
    #     print(f"🛑 Token(s) con índice negativo detectados:")
    #     for i in idx:
    #         print(f" - Posición {tuple(i.tolist())}, valor: {tokens[tuple(i.tolist())].item()}")
    #     raise ValueError("Se encontraron índices negativos en los tokens.")


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


class DelphiData:
    def __init__(self, data_dir, val_fold, delphi_labels, labels, device=None):
        self.data_dir = data_dir
        self.delphi_labels = pd.read_csv(delphi_labels)
        self.labels = pd.read_csv(labels, header=None, sep="\t")
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.val_fold = val_fold

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

    def tokens_to_ids(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[int(id_)] for id_ in ids]

    def split_person(self, p):
        tokens = [i[0] for i in p]
        ages = [i[1] for i in p]
        return tokens, ages
    

def get_val_data(val_filename):
    val_data = np.memmap(val_filename, dtype=np.int32).reshape(-1, 3)
    val_p2i = get_p2i(val_data)
    return val_data, val_p2i


def fix_artifact_uri(artifact_uri):
    artifact_uri = re.sub(pattern="^file://", repl="", string=artifact_uri)
    artifact_uri = re.sub(pattern=".*/mlruns", repl="mlruns", string=artifact_uri)
    import pathlib
    artifact_uri = pathlib.Path(artifact_uri)
    return artifact_uri


def get_epoch_from_ckpt(ckpt_path):
    return int(ckpt_path.split("_")[-1].split(".")[0])


def get_ignored_tokens(runinfo, validation_loss_mode = True):
    """
    Get the list of ignored tokens from the runinfo.
    """
    ignored_tokens = ast.literal_eval(runinfo['ignore_tokens'])
    if validation_loss_mode:
        ignored_tokens += [NO_EVENT_TOKEN_ID]    
    
    if isinstance(ignored_tokens, int):
        ignored_tokens = [ignored_tokens]
    return ignored_tokens


def get_top_counts(data, labels, top_n=200, ignored_tokens=[]):

    id_to_token = dict(zip(labels.index-1, labels.name))

    counts = pd.DataFrame(data, columns=["subject_id", "age", "token_id"]).\
        query("token_id not in @ignored_tokens").\
        assign(token=lambda df: df.token_id.apply(lambda x: id_to_token[x])).\
        token.value_counts(ascending=False).\
        head(top_n).\
        sort_values()
    
    return counts


def get_wte(model):
   wte = model.transformer.wte.weight.detach().numpy()
   return pd.DataFrame(wte, index=[ id_to_token[i] for i in range(-1, len(id_to_token)-1) ])


def get_best_ckpt(runinfo):
    """
    Get the path to the best checkpoint from the runinfo.
    """
    ckpt_dir = fix_artifact_uri(runinfo.artifact_uri) / "checkpoints"
    best_ckpt_path = ckpt_dir / sorted(os.listdir(ckpt_dir), key=get_epoch_from_ckpt)[-1]
    return best_ckpt_path
