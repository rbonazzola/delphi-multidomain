import mlflow
import re
import os
import numpy as np

from utils import get_p2i

from typing import List, Dict, Tuple, Optional, Union

DATA_TYPE_CONFIGS = {

    'real-hla-4digits-5folds': {
        'labels': './data/ukb_real_5_folds_4digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla4d.csv',
        'data_root': './data/ukb_real_5_folds_4digit',
    },
    'real-nohla-5folds': {
        'labels': './data/ukb_real_5_folds_nohla/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd.csv',
        'data_root': './data/ukb_real_5_folds_nohla',
    },
    'real-hla-2digits-5folds': {
        'labels': './data/ukb_real_5_folds_2digit/labels.csv',
        'delphi_labels': 'delphi_labels_chapters_colours_icd_with_hla2d.csv',
        'data_root': './data/ukb_real_5_folds_2digit',
    },
}


def get_best_ckpt_from_mlflow(runid):
    runinfo = mlflow.get_run(run_id=runid)
    ckpt_dir = re.sub(r'^.*(?=mlruns)', '', runinfo.info.artifact_uri) + '/checkpoints'
    best_ckpt = [x for x in os.listdir(ckpt_dir) if 'best' in x][0]
    best_ckpt = os.path.join(ckpt_dir, best_ckpt)
    return best_ckpt


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


def get_data_partitions(datafile, fold):

    '''
    datafile: numpy file containing three columns (subject_id, time, token_id)
    '''

    load_data_from_bin = lambda file: np.fromfile(file, dtype=np.uint32).reshape(-1, 3)

    data = load_data_from_bin(datafile)
    fold_ids = load_fold_ids("data/transforms/subject_lists", num_folds=10)

    splits = generate_splits(fold_ids, n_train_folds=7, n_val_folds=1, n_test_folds=2, val_as_last=True)  
    split_idx = fold - 1
    
    train_ids = splits[split_idx]["train"]
    val_ids   = splits[split_idx]["valid"]
    test_ids  = splits[split_idx]["test"]

    train_data = data[np.isin(data[:,0], train_ids)]
    val_data   = data[np.isin(data[:,0], val_ids)]
    test_data  = data[np.isin(data[:,0], test_ids)]

    train_p2i  = get_p2i(train_data)
    val_p2i    = get_p2i(val_data)
    test_p2i   = get_p2i(test_data)

    return (train_data, train_p2i, train_ids), \
           (val_data, val_p2i, val_ids), \
           (test_data, test_p2i, test_ids)