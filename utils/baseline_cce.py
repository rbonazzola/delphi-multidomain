"""Age-sex stratified disease-incidence baseline for relative CCE evaluation."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Left edges of age bins in years; the last bin captures everything >= 80.
AGE_BINS_YEARS = [0, 20, 30, 40, 50, 60, 70, 80]
AGE_BIN_LABELS = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
SEX_LABELS = ["Female", "Male"]


class BaselineCCECalculator:
    """
    Computes cross-entropy against an age-sex stratified disease incidence baseline.

    The incidence file (created by auc/compute_disease_incidence.py) is a parquet with
    columns: sex (int 0/1), age_bin (int), token_id (int), prob (float).
    """

    def __init__(self, incidence_path: str, disease_vocab_size: int):
        df = pd.read_parquet(incidence_path)
        n_bins = len(AGE_BINS_YEARS)

        table = np.zeros((2, n_bins, disease_vocab_size), dtype=np.float32)
        for row in df.itertuples(index=False):
            table[int(row.sex), int(row.age_bin), int(row.token_id)] = row.prob

        self.table = torch.tensor(table)   # [2, n_bins, vocab_size]
        self.n_bins = n_bins
        self.disease_vocab_size = disease_vocab_size

    def age_to_bin(self, ages_days: torch.Tensor) -> torch.Tensor:
        boundaries = torch.tensor(AGE_BINS_YEARS[1:], dtype=torch.float32, device=ages_days.device)
        return torch.bucketize(ages_days.float() / 365.25, boundaries)

    def compute(
        self,
        f_local_ids: torch.Tensor,  # [N] local disease token IDs
        f_ages_days: torch.Tensor,  # [N] ages in days at each prediction step
        f_sex: torch.Tensor,        # [N] int 0=Female 1=Male
    ):
        """
        Returns:
            baseline_cce: scalar mean NLL under the incidence baseline
            per_token_nll: Tensor [N]
            age_bins: LongTensor [N]
        """
        age_bins = self.age_to_bin(f_ages_days)
        table = self.table.to(f_local_ids.device)
        probs = table[f_sex.long(), age_bins, f_local_ids.long()]
        nll = -torch.log(probs.clamp(min=1e-10))
        return nll.mean(), nll, age_bins
