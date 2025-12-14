# card_model.py

import torch
import torch.nn as nn

class CardPolicyNetwork(nn.Module):
    def __init__(self, input_dim=127, hidden_dim=512, output_dim=36):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
