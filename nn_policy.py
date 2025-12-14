# nn_policy.py
import torch
import torch.nn as nn
import numpy as np

from nn_state import NNState   # from Step A


class NNCardPolicy:
    """
    Wrapper for your trained card model.
    Provides:
       - masked softmax policy
       - raw logits for MCTS
    """

    def __init__(self, model_class, model_path: str, device: torch.device):
        """
        model_class: the class of your card network, e.g., CardNet
        model_path:  path to card_model.pt
        """
        self.device = device

        # Create and load model
        self.model = model_class()
        sd = torch.load(model_path, map_location=device)
        self.model.load_state_dict(sd)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def forward_logits(self, state: NNState) -> np.ndarray:
        """
        Runs the model on the given state → returns raw logits (36,)
        NOTE: does NOT apply legal mask.
        """
        x, _ = state.to_torch(self.device)  # x: (1,127), legal_mask: (1,36)
        logits = self.model(x)              # (1,36)
        return logits[0].cpu().numpy()

    @torch.no_grad()
    def policy(self, state: NNState) -> np.ndarray:
        """
        Returns masked softmax policy over 36 cards.
        """
        x, legal_mask = state.to_torch(self.device)  # shapes: (1,127), (1,36)

        logits = self.model(x)[0]  # (36,)
        legal_mask = legal_mask[0] # (36,)

        # Kill illegal moves
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = -1e9

        # Softmax on masked logits
        probs = torch.softmax(masked_logits, dim=-1)

        return probs.cpu().numpy()

    @torch.no_grad()
    def greedy_action(self, state: NNState) -> int:
        """
        Returns the NN greedy move (argmax over masked probabilities).
        Useful for debugging.
        """
        probs = self.policy(state)
        return int(np.argmax(probs))
