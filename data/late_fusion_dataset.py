"""
LateFusion data layer: wraps DelphiDataset/CollateFn with per-subject HLA features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch.utils.data import Dataset

from data.dataset import DelphiBatch, DelphiCollateFn


@dataclass
class LateFusionBatch:
    """
    DelphiBatch extended with a precomputed per-subject HLA feature vector.

    Transparent to Trainer / shared_step: attribute lookups for standard
    DelphiBatch fields (global_token_ids, domain_ids, ages, eval_mask, ...)
    are delegated to delphi_batch via __getattr__.
    """
    delphi_batch: DelphiBatch
    hla_features: torch.Tensor   # [B, d_hla]

    def __getattr__(self, name: str):
        # Guard against recursion during pickling / before fields are initialised.
        if name in ("delphi_batch", "hla_features"):
            raise AttributeError(name)
        return getattr(self.delphi_batch, name)

    def to(self, device: Union[str, torch.device]) -> "LateFusionBatch":
        return LateFusionBatch(
            delphi_batch=self.delphi_batch.to(device),
            hla_features=self.hla_features.to(device),
        )

    @property
    def batch_size(self) -> int:
        return self.delphi_batch.batch_size

    @property
    def seq_len(self) -> int:
        return self.delphi_batch.seq_len

    def __repr__(self) -> str:
        return (
            f"LateFusionBatch(B={self.batch_size}, T={self.seq_len}, "
            f"d_hla={self.hla_features.shape[-1]})"
        )


class LateFusionDataset(Dataset):
    """
    Wraps a DelphiDataset and appends a precomputed HLA feature vector per subject.

    __getitem__ returns (delphi_item_dict, hla_feat_tensor) so that
    LateFusionCollateFn can separate and combine them.
    """

    def __init__(self, delphi_dataset, hla_features: torch.Tensor):
        if len(delphi_dataset) != len(hla_features):
            raise ValueError(
                f"Dataset length {len(delphi_dataset)} != "
                f"HLA features length {len(hla_features)}"
            )
        self.delphi_dataset = delphi_dataset
        self.hla_features = hla_features   # [N, d_hla]

    def __len__(self) -> int:
        return len(self.delphi_dataset)

    def __getitem__(self, idx):
        return self.delphi_dataset[idx], self.hla_features[idx].clone()

    @property
    def _subject_ids(self):
        return self.delphi_dataset._subject_ids

    @property
    def subject_list(self):
        return self.delphi_dataset._subject_ids.tolist()


class LateFusionCollateFn:
    """
    Wraps DelphiCollateFn; additionally stacks HLA feature vectors into a LateFusionBatch.
    """

    def __init__(self, delphi_collate_fn: DelphiCollateFn):
        self.delphi_collate = delphi_collate_fn

    def __call__(self, batch) -> LateFusionBatch:
        seq_items = [item[0] for item in batch]
        hla_feats = torch.stack([item[1] for item in batch])   # [B, d_hla]
        delphi_batch = self.delphi_collate(seq_items)
        return LateFusionBatch(delphi_batch=delphi_batch, hla_features=hla_feats)
