#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import os

def load_bin(file):
    """Load a memmap file and report its content summary."""
    array = np.memmap(file, dtype=np.int32)
    n_subjects = len(np.unique(array.reshape(-1, 3)))
    n_tokens = len(array) // 3
    print(f"[INFO] Loaded {os.path.basename(file)} "
          f"→ {n_subjects} subjects, {n_tokens} tokens")
    return array

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple memmap files of int32 triplets, "
                    "sort by subject and token, and save as .npy"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input memmap files (.bin, .npy, etc.). Provide n-1 files."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file (.npy) to save concatenated array."
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("You must provide at least two input files.")

    arrays = []
    for f in args.inputs:
        arrays.append(load_bin(f))

    print("[INFO] Concatenating arrays...")
    all_array = np.concatenate(arrays)
    all_array = (
        pd.DataFrame(all_array.reshape(-1, 3))
        .sort_values([0, 1])
        .values
    )

    np.save(args.output, all_array)
    print(f"[INFO] Saved concatenated array to {args.output}")

if __name__ == "__main__":
    main()
