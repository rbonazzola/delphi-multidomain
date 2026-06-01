import itertools
from pathlib import Path


def load_fold_ids(fold_dir: str, num_folds: int = 10) -> list[list[str]]:
    """
    Load subject IDs from predefined fold files.
    Each file must be named `fold_i_of_num.txt` (0-indexed).
    """
    folds = []
    for i in range(1, num_folds + 1):
        fname = Path(fold_dir) / f"subset{i}of{num_folds}.csv"
        with fname.open() as f:
            ids = [line.strip() for line in f if line.strip()]
            folds.append(ids)
    return folds


def generate_splits(
    folds: list[list[str]],
    n_train_folds: int,
    n_val_folds: int,
    n_test_folds: int,
) -> list[dict[str, list[str]]]:

    num_folds = len(folds)

    splits = []

    for start in range(0, num_folds, n_test_folds):
        test_idx = [(start + i) % num_folds for i in range(n_test_folds)]
        val_idx = [(start - 1) % num_folds]

        train_idx = [i for i in range(num_folds) if i not in test_idx and i not in val_idx]

        split = {
            "train": list(itertools.chain.from_iterable(folds[i] for i in train_idx)),
            "valid": list(itertools.chain.from_iterable(folds[i] for i in val_idx)),
            "test": list(itertools.chain.from_iterable(folds[i] for i in test_idx)),
        }

        splits.append(split)

    return splits


def get_data_partitions(folder, fold):
    """
    datafile: numpy file containing three columns (subject_id, time, token_id)
    """

    fold_ids = load_fold_ids(folder, num_folds=10)

    splits = generate_splits(fold_ids, n_train_folds=7, n_val_folds=1, n_test_folds=2)
    split_idx = fold - 1

    train_ids = [str(x) for x in splits[split_idx]["train"]]
    val_ids = [str(x) for x in splits[split_idx]["valid"]]
    test_ids = [str(x) for x in splits[split_idx]["test"]]

    return train_ids, val_ids, test_ids
