# trump_dataset.py

import torch
from torch.utils.data import Dataset
from typing import Any


class TrumpDataset(Dataset):
    """
    PyTorch dataset for Jass trump prediction.

    Expects features as a two dimensional array like object
    and labels as a one dimensional array like object.
    """

    def __init__(self, features: Any, labels: Any) -> None:
        # float32 for neural network input
        self.X = torch.tensor(features, dtype=torch.float32)
        # long for class indices
        self.y = torch.tensor(labels, dtype=torch.long)

        assert len(self.X) == len(self.y), "Features and labels must have the same length"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
