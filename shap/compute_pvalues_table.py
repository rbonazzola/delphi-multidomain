"""
Aggregate delta-logit .pkl files produced by individual sarray_params jobs
(one per allele) into a single summary TSV.

Each pkl is named:
    {disease_id}__{allele_id}__{sex_label}.pkl

Usage:
    python shap/compute_pvalues_table.py \\
        --pkl_dir shap/output_delta_logit \\
        --output  shap/output_delta_logit/summary_single.tsv \\
        [--min_n 10] [--sex both_sexes]

Arguments:
    --pkl_dir   Directory containing the .pkl files (searched recursively).
    --output    Output CSV path (default: <pkl_dir>/summary_single.csv).
    --min_n     Minimum number of delta values to compute Wilcoxon (default: 10).
    --sex       If given, restrict to pkl files with this sex label (e.g. both_sexes).
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from tqdm import tqdm

DELPHI_DIR = Path("/nfs/research/birney/users/bonazzola/repos/delphis/delphi-refactor")

AGE_BRACKETS = ["0-20", "10-30", "20-40", "30-50", "40-60", "50-70", "60-80"]

# Filename pattern: {disease_id}__{allele_id}__{sex_label}.pkl
_PKL_RE = re.compile(r"^(\d+)__(\d+)__(.+)\.pkl$")


def load_tokenizer(path: Path) -> list:
    return list(yaml.safe_load(open(path)))


def process_pkl(
    pkl_path: Path,
    disease_tok: list,
    hla_tok: list,
    min_n: int,
) -> dict | None:
    m = _PKL_RE.match(pkl_path.name)
    if not m:
        return None

    disease_id = int(m.group(1))
    allele_id  = int(m.group(2))
    sex_label  = m.group(3)

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  SKIP (load error): {pkl_path} — {e}", file=sys.stderr)
        return None

    delta        = np.asarray(data["delta"]).ravel()
    age_brackets = np.asarray(data.get("age_brackets", []), dtype=object).ravel()
    n = len(delta)

    if n < min_n:
        stat = float("nan")
        p    = float("nan")
    else:
        try:
            res  = stats.wilcoxon(delta)
            stat = float(res.statistic)
            p    = float(res.pvalue)
        except Exception:
            stat = float("nan")
            p    = float("nan")

    disease_name = (
        disease_tok[disease_id] if disease_id < len(disease_tok) else str(disease_id)
    )
    allele_name = (
        hla_tok[allele_id] if allele_id < len(hla_tok) else str(allele_id)
    )

    row = {
        "disease_id":    disease_id,
        "disease_name":  disease_name,
        "allele_id":     allele_id,
        "allele_name":   allele_name,
        "sex":           sex_label,
        "n_subjects":    n,
        "mean_delta":    float(np.mean(delta))   if n > 0 else float("nan"),
        "median_delta":  float(np.median(delta)) if n > 0 else float("nan"),
        "wilcoxon_stat": stat,
        "p_value":       p,
    }

    for bracket in AGE_BRACKETS:
        mask = np.array([b == bracket for b in age_brackets])
        sub  = delta[mask]
        col  = bracket.replace("-", "_")
        if len(sub) < min_n:
            bp = float("nan")
        else:
            try:
                bp = float(stats.wilcoxon(sub).pvalue)
            except Exception:
                bp = float("nan")
        row[f"n_{col}"]          = int(mask.sum())
        row[f"mean_delta_{col}"] = float(np.mean(sub)) if len(sub) else float("nan")
        row[f"p_{col}"]          = bp

    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pkl_dir", type=Path, required=True,
                        help="Directory containing {disease_id}__{allele_id}__{sex}.pkl files.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: <pkl_dir>/summary_single.csv).")
    parser.add_argument("--min_n", type=int, default=10,
                        help="Minimum delta count for Wilcoxon test (default: 10).")
    parser.add_argument("--sex", type=str, default=None,
                        help="Restrict to this sex label (e.g. both_sexes).")
    args = parser.parse_args()

    disease_tok = load_tokenizer(
        DELPHI_DIR / "data/transforms/tokens/diseases/tokenizer.yaml"
    )
    hla_tok = load_tokenizer(
        DELPHI_DIR / "data/transforms/tokens/hla_alleles/tokenizer.yaml"
    )

    pkl_files = sorted(args.pkl_dir.rglob("*.pkl"))
    pkl_files = [p for p in pkl_files if _PKL_RE.match(p.name)]
    if args.sex:
        pkl_files = [p for p in pkl_files if f"__{args.sex}.pkl" in p.name]

    print(f"Found {len(pkl_files)} pkl files matching single-allele pattern.")
    if not pkl_files:
        print("No files to process. Exiting.")
        return

    rows = []
    for pkl_path in tqdm(pkl_files, desc="Processing pkl files"):
        row = process_pkl(pkl_path, disease_tok, hla_tok, args.min_n)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid rows produced.")
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(
            ["disease_id", "sex", "mean_delta"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

    output = args.output or args.pkl_dir / "summary_single.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows → {output}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
