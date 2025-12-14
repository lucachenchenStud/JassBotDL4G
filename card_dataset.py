# card_dataset.py

import torch
from torch.utils.data import Dataset
from state_encoder import encode_sample


class CardPlayingDataset(Dataset):
    """
    PyTorch Dataset wrapping all card-play samples.
    Converts raw sample dicts into encoded tensors:
        state_vector: [127]
        legal_mask:   [36]
        action_index: scalar
    """

    def __init__(self, samples):
        """
        samples: list of dicts loaded from all_samples_raw.pt
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Use your encoder
        state_vector, legal_mask, action_index = encode_sample(sample)

        # Convert to tensors
        state_vector = torch.tensor(state_vector, dtype=torch.float32)
        legal_mask   = torch.tensor(legal_mask, dtype=torch.float32)
        action_index = torch.tensor(action_index, dtype=torch.long)

        return state_vector, legal_mask, action_index
