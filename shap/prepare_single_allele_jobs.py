"""
Generate a sarray_params TSV for single-allele SHAP jobs (custom_hla_shap.py),
covering all (disease_id, allele_id, sex) combinations with enough cases.

For each combination, "cases" = subjects who carry the allele AND have the
disease, restricted to genetic_white_ids.

Output TSV columns: disease_id, allele_id, sex, output
  - sex is "male", "female", or omitted (empty = no sex filter = both)
  - output is a path template resolved by custom_hla_shap.py at runtime

Usage:
    python shap/prepare_single_allele_jobs.py \\
        --min_count 10 \\
        --output shap/single_allele_jobs.tsv
"""

import argparse
from pathlib import Path

import pandas as pd

DELPHI_DIR = Path(__file__).resolve().parent.parent
DATA_DIR   = DELPHI_DIR / "data/transforms/tokens"

SUBJECTS_FILE = DELPHI_DIR / "data/transforms/subject_lists/genetic_white_ids.txt"
OUTPUT_TEMPLATE = "shap/output_delta_logit/{disease_id}__{allele_id}__{sex}.pkl"


def load_subjects(path):
    return set(pd.read_csv(path, header=None)[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min_count", type=int, default=10,
                        help="Minimum number of cases per (disease, allele, sex) (default: 10)")
    parser.add_argument("--output", type=str, default="shap/single_allele_jobs.tsv",
                        help="Output TSV path (default: shap/single_allele_jobs.tsv)")
    parser.add_argument("--subjects", type=str, default=str(SUBJECTS_FILE),
                        help="Path to subject ID list (default: genetic_white_ids.txt)")
    parser.add_argument("--sexes", nargs="+", default=["both", "male", "female"],
                        choices=["both", "male", "female"],
                        help="Which sex strata to include (default: all three)")
    args = parser.parse_args()

    print("Loading subject list...")
    white_ids = load_subjects(args.subjects)

    print("Loading sex tokens...")
    sex_df = pd.read_csv(DATA_DIR / "sex/tokens.csv")
    sex_df = sex_df[sex_df["subject_id"].isin(white_ids)][["subject_id", "token_id"]]
    sex_df = sex_df.rename(columns={"token_id": "sex_id"})
    sex_df = sex_df.drop_duplicates("subject_id")  # one sex per subject

    print("Loading HLA tokens...")
    hla_df = pd.read_csv(DATA_DIR / "hla_alleles/tokens.csv")[["subject_id", "token_id"]]
    hla_df = hla_df[hla_df["subject_id"].isin(white_ids)]
    hla_df = hla_df.drop_duplicates(["subject_id", "token_id"])
    hla_df = hla_df.rename(columns={"token_id": "allele_id"})

    print("Loading disease tokens...")
    dis_df = pd.read_csv(DATA_DIR / "diseases/tokens.csv")[["subject_id", "token_id"]]
    dis_df = dis_df[dis_df["subject_id"].isin(white_ids)]
    dis_df = dis_df.drop_duplicates(["subject_id", "token_id"])
    dis_df = dis_df.rename(columns={"token_id": "disease_id"})

    print("Computing co-occurrences (allele × disease)...")
    # subjects with allele AND disease
    merged = hla_df.merge(dis_df, on="subject_id")
    # add sex
    merged = merged.merge(sex_df, on="subject_id", how="left")
    # sex_id: 0=female, 1=male (from sex tokenizer)

    rows = []

    if "both" in args.sexes:
        counts_both = (
            merged.groupby(["allele_id", "disease_id"])["subject_id"]
            .nunique()
            .reset_index(name="n")
        )
        counts_both = counts_both[counts_both["n"] >= args.min_count]
        counts_both["sex"] = "both"
        rows.append(counts_both)

    if "male" in args.sexes:
        counts_male = (
            merged[merged["sex_id"] == 1]
            .groupby(["allele_id", "disease_id"])["subject_id"]
            .nunique()
            .reset_index(name="n")
        )
        counts_male = counts_male[counts_male["n"] >= args.min_count]
        counts_male["sex"] = "male"
        rows.append(counts_male)

    if "female" in args.sexes:
        counts_female = (
            merged[merged["sex_id"] == 0]
            .groupby(["allele_id", "disease_id"])["subject_id"]
            .nunique()
            .reset_index(name="n")
        )
        counts_female = counts_female[counts_female["n"] >= args.min_count]
        counts_female["sex"] = "female"
        rows.append(counts_female)

    result = pd.concat(rows, ignore_index=True)
    result["output"] = OUTPUT_TEMPLATE
    result = result[["disease_id", "allele_id", "sex", "output"]].sort_values(
        ["disease_id", "allele_id", "sex"]
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)

    print(f"\n{len(result)} jobs after filtering (min_count={args.min_count})")
    print(f"  both sexes : {(result['sex'] == 'both').sum()}")
    print(f"  male only  : {(result['sex'] == 'male').sum()}")
    print(f"  female only: {(result['sex'] == 'female').sum()}")
    print(f"\nSaved → {out_path}")
    print(f"\nTo submit:")
    print(f"  source ~/repos/codon_helpers/slurm_functions.sh")
    print(f"  sarray_params shap/custom_hla_shap.py {out_path} \\")
    print(f"    --experiment_id 263078128312970150 \\")
    print(f"    --subjects {args.subjects} \\")
    print(f"    --n_counterfactuals 5 \\")
    print(f"    --run_name hla_mix --param n_head=12 \\")
    print(f"    --time=06:00:00 --mem=32G --cpus=4")


if __name__ == "__main__":
    main()
