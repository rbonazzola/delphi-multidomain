"""
Generate token_metadata.tsv from an HLA tokenizer.yaml.

Parses allele names like "HLA-A*01:01" to extract:
  - locus:          hla_a
  - locus_group:    hla_a  (groups DPA/DPB → hla_dp, DQA/DQB → hla_dq)
  - allele_1field:  HLA-A*01
  - allele_2field:  HLA-A*01:01  (same as name for 2-field entries)

Usage:
    python generate_hla_metadata.py [--tokenizer_path PATH] [--output_path PATH]

Defaults:
    --tokenizer_path  data/transforms/tokens/hla_alleles/tokenizer.yaml
    --output_path     data/transforms/tokens/hla_alleles/token_metadata.tsv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml


def parse_allele(name: str) -> dict:
    """
    Parse an HLA allele name into its components.
    
    Examples:
        "HLA-A*01:01"   -> locus=hla_a,   1field=HLA-A*01,  2field=HLA-A*01:01
        "HLA-DPA1*01:03" -> locus=hla_dpa, 1field=HLA-DPA1*01, 2field=HLA-DPA1*01:03
        "HLA-DRB1*03:01" -> locus=hla_drb, 1field=HLA-DRB1*03, 2field=HLA-DRB1*03:01
    """
    result = {"name": name}

    # Match pattern: HLA-<GENE>*<field1>:<field2>[:...]
    match = re.match(r"^(HLA-\w+)\*(.+)$", name)
    if not match:
        # Fallback for non-standard names
        result["locus"] = "unknown"
        result["allele_1field"] = name
        result["allele_2field"] = name
        return result

    gene = match.group(1)       # e.g. "HLA-A", "HLA-DPA1", "HLA-DRB1"
    fields = match.group(2)     # e.g. "01:01", "03:01:01"

    # Locus: lowercase, strip trailing digits from gene name
    # HLA-A -> hla_a, HLA-DPA1 -> hla_dpa, HLA-DRB1 -> hla_drb
    gene_stripped = re.sub(r"\d+$", "", gene)  # HLA-DPA1 -> HLA-DPA
    locus = gene_stripped.lower().replace("-", "_")  # hla_dpa

    # 1-field: gene + first field
    field_parts = fields.split(":")
    allele_1field = f"{gene}*{field_parts[0]}"

    # 2-field: gene + first two fields (if available)
    if len(field_parts) >= 2:
        allele_2field = f"{gene}*{field_parts[0]}:{field_parts[1]}"
    else:
        allele_2field = allele_1field

    # locus_group: collapse DPA/DPB → hla_dp, DQA/DQB → hla_dq, rest unchanged
    locus_group = re.sub(r"^(hla_d[pq])[ab]$", r"\1", locus)

    result["locus"] = locus
    result["locus_group"] = locus_group
    result["allele_1field"] = allele_1field
    result["allele_2field"] = allele_2field

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate HLA token metadata from tokenizer.yaml")
    parser.add_argument(
        "--tokenizer_path",
        default="data/transforms/tokens/hla_alleles/tokenizer.yaml",
    )
    parser.add_argument(
        "--output_path",
        default="data/transforms/tokens/hla_alleles/token_metadata.csv",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path)
    output_path = Path(args.output_path)

    # Load tokenizer
    with open(tokenizer_path) as f:
        tokenizer = yaml.safe_load(f)

    print(f"Loaded {len(tokenizer)} alleles from {tokenizer_path}")

    # Parse each allele
    rows = []
    for token_id, allele_name in enumerate(tokenizer):
        row = parse_allele(allele_name)
        row["token_id"] = token_id
        rows.append(row)

    df = pd.DataFrame(rows)[["token_id", "name", "locus", "locus_group", "allele_1field", "allele_2field"]]

    # Summary
    print(f"\nLocus distribution:")
    print(df["locus"].value_counts().to_string())
    print(f"\nLocus group distribution:")
    print(df["locus_group"].value_counts().to_string())
    print(f"\nTotal 1-field groups: {df['allele_1field'].nunique()}")
    print(f"Total 2-field groups: {df['allele_2field'].nunique()}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
