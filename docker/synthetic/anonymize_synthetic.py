"""
Prepares a synthetic cohort (output of generate_synthetic_subjects.py) for
distribution inside the Docker image.

The synthetic cohort is not fully synthetic: generate_synthetic_subjects.py
copies the real sex and the real lifestyle tokens (with their real ages) of
the UKB subjects used for conditioning. Only diseases, death, and their ages
are sampled by the model. This script reduces the re-identification risk of
the copied parts:

1. Lifestyle ages: each (subject, visit) gets one shift drawn from
   N(0, sigma_years). All tokens of the same visit (bmi/smoking/alcohol
   recorded at the same age) get the same shift, so they stay together.
   Shifted ages are rounded to whole days and clipped to [0, age at death].
2. Subject IDs: remapped by a random permutation, consistently across
   domains. This breaks the order link between synthetic IDs and the real
   conditioning subject IDs.
3. generation_metadata.json is NOT copied: it lists the real conditioning
   subject IDs.

Usage:
    python docker/synthetic/anonymize_synthetic.py \\
        --input_dir data/transforms/tokens/synthetic \\
        --output_dir /tmp/synthetic_anon \\
        --sigma_years 2.0 --seed 2026
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ("sex", "lifestyle", "diseases", "death")
DAYS_PER_YEAR = 365.25


def jitter_lifestyle(lifestyle: pd.DataFrame, death: pd.DataFrame, sigma_years: float,
                     rng: np.random.Generator) -> pd.DataFrame:
    out = lifestyle.copy()
    if out.empty:
        return out
    visit = out.groupby(["subject_id", "age"]).ngroup().to_numpy()
    shifts = rng.normal(0.0, sigma_years * DAYS_PER_YEAR, size=visit.max() + 1)
    new_age = np.round(out["age"].to_numpy(dtype=float) + shifts[visit])

    death_age = death.groupby("subject_id")["age"].min()
    upper = out["subject_id"].map(death_age).fillna(np.inf).to_numpy(dtype=float)
    out["age"] = np.clip(new_age, 0.0, upper).astype(np.int64)
    return out


def remap_ids(frames: dict, id_offset: int, rng: np.random.Generator) -> dict:
    all_ids = np.unique(np.concatenate([df["subject_id"].to_numpy() for df in frames.values()]))
    new_ids = id_offset + rng.permutation(len(all_ids))
    mapping = pd.Series(new_ids, index=all_ids)
    return {d: df.assign(subject_id=df["subject_id"].map(mapping).to_numpy()) for d, df in frames.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--sigma_years", type=float, default=2.0,
                   help="Standard deviation (years) of the Gaussian shift on lifestyle ages")
    p.add_argument("--seed", type=int, required=True,
                   help="RNG seed. Do not distribute it: with the seed, the shifts can be reproduced.")
    p.add_argument("--id_offset", type=int, default=9_000_000)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")

    rng = np.random.default_rng(args.seed)
    frames = {d: pd.read_csv(args.input_dir / d / "tokens.csv") for d in DOMAINS}

    orig_lifestyle = frames["lifestyle"]
    frames["lifestyle"] = jitter_lifestyle(orig_lifestyle, frames["death"], args.sigma_years, rng)
    if len(orig_lifestyle):
        delta_years = (frames["lifestyle"]["age"].to_numpy() - orig_lifestyle["age"].to_numpy()) / DAYS_PER_YEAR
        print(f"lifestyle shift (years): mean={delta_years.mean():+.3f} sd={delta_years.std():.3f} "
              f"|shift|<1 day: {(np.abs(delta_years) * DAYS_PER_YEAR < 1).mean():.2%}")

    frames = remap_ids(frames, args.id_offset, rng)

    for d, df in frames.items():
        dst = args.output_dir / d
        dst.mkdir(parents=True, exist_ok=True)
        df.sort_values(["subject_id", "age", "token_id"]).to_csv(dst / "tokens.csv", index=False)
        shutil.copyfile(args.input_dir / d / "tokenizer.yaml", dst / "tokenizer.yaml")
        print(f"{d:10s} rows={len(df):8d} subjects={df['subject_id'].nunique()}")

    print(f"Wrote anonymized cohort to {args.output_dir} (generation_metadata.json omitted)")


if __name__ == "__main__":
    main()
