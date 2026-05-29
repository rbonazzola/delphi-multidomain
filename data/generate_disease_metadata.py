"""
Generate token_metadata.tsv from a disease tokenizer.yaml and tokens.csv.

Parses entries like "A00 Cholera" to extract:
  - icd_code:        A00
  - icd_description: Cholera

Also computes from tokens.csv:
  - n_subjects:  number of subjects with each token (assumes at most one event per subject per code)

Usage:
    python generate_disease_metadata.py [--tokenizer_path PATH] [--tokens_path PATH] [--output_path PATH]

Defaults:
    --tokenizer_path  data/transforms/tokens/diseases/tokenizer.yaml
    --tokens_path     data/transforms/tokens/diseases/tokens.csv
    --output_path     data/transforms/tokens/diseases/token_metadata.tsv
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_icd_entry(name: str) -> dict:
    """
    Parse a disease token entry into ICD code and description.

    Examples:
        "A00 Cholera"                    -> icd_code=A00, icd_description=Cholera
        "B20 HIV disease"                -> icd_code=B20, icd_description=HIV disease
        "Z99 something"                  -> icd_code=Z99, icd_description=something
    """
    parts = name.split(" ", 1)
    if len(parts) == 2:
        return {"name": name, "icd_code": parts[0], "icd_description": parts[1]}
    else:
        return {"name": name, "icd_code": name, "icd_description": ""}


def main():
    parser = argparse.ArgumentParser(description="Generate disease token metadata")
    parser.add_argument(
        "--tokenizer_path",
        default="data/transforms/tokens/diseases/tokenizer.yaml",
    )
    parser.add_argument(
        "--tokens_path",
        default="data/transforms/tokens/diseases/tokens.csv",
    )
    parser.add_argument(
        "--output_path",
        default="data/transforms/tokens/diseases/token_metadata.tsv",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path)
    tokens_path = Path(args.tokens_path)
    output_path = Path(args.output_path)

    # Load tokenizer
    with tokenizer_path.open() as f:
        tokenizer = yaml.safe_load(f)

    print(f"Loaded {len(tokenizer)} tokens from {tokenizer_path}")

    # Parse each entry
    rows = []
    for token_id, name in enumerate(tokenizer):
        row = parse_icd_entry(name)
        row["token_id"] = token_id
        rows.append(row)

    df = pd.DataFrame(rows)[["token_id", "name", "icd_code", "icd_description"]]

    # Compute per-token subject and event counts from tokens.csv
    if tokens_path.exists():
        tokens = pd.read_csv(tokens_path)
        n_subjects = tokens.groupby("token_id")["subject_id"].nunique().rename("n_subjects").reset_index()
        df = df.merge(n_subjects, on="token_id", how="left")
        df["n_subjects"] = df["n_subjects"].fillna(0).astype(int)
        print(f"Loaded event counts from {tokens_path}")
    else:
        print(f"Warning: {tokens_path} not found, skipping subject/event counts")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep="\t")
    print(f"\nSaved to {output_path}")
    print(f"\nTotal tokens: {len(df)}")
    print(f"Tokens with no subjects: {(df['n_subjects'] == 0).sum()}")
    print("\nTop 10 by n_subjects:")
    print(df.nlargest(10, "n_subjects")[["icd_code", "icd_description", "n_subjects"]].to_string(index=False))


if __name__ == "__main__":
    main()
