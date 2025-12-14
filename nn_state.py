from dataclasses import dataclass

import numpy as np
import torch

from jass.game.game_observation import GameObservation
from jass.game.const import card_ids
from state_encoder import CARD_TO_INDEX, CARD_LIST, encode_observation

from jass.game.rule_schieber import RuleSchieber

RULE = RuleSchieber()


# Map jass-kit card_ids → your neural-network indices
INT_TO_NN_INDEX = {
    jass_id: CARD_TO_INDEX[str_card]
    for str_card, jass_id in card_ids.items()
}

NUM_CARDS = 36

TRUMP_MAP = {
    0: "DIAMONDS",
    1: "HEARTS",
    2: "SPADES",
    3: "CLUBS",
    4: "OBE_ABE",
    5: "UNE_UFE",
    6: "PUSH",
}


def cards_mask_from_int_ids(int_ids):
    """
    Convert jass-kit card integers (0..35 but in custom order)
    into NN 36-dim one-hot mask.
    """
    mask = np.zeros(NUM_CARDS, dtype=np.float32)
    for cid in int_ids:
        if cid != -1:  # -1 = empty
            nn_index = INT_TO_NN_INDEX[cid]
            mask[nn_index] = 1.0
    return mask


def compute_legal_cards(obs: GameObservation):
    """
    Compute legal moves directly from official Jass rules.
    Returns a (36,) float32 mask in YOUR neural network card order.
    """
    # jass-kit one-hot mask in its internal order
    legal_jass = RULE.get_valid_cards_from_obs(obs)  # np array (36,)

    legal_mask = np.zeros(36, dtype=np.float32)

    # Convert from jass-kit indexing → your NN indexing
    for jass_id, is_legal in enumerate(legal_jass):
        if is_legal:
            nn_idx = INT_TO_NN_INDEX[jass_id]  # <-- mapping from your earlier code
            legal_mask[nn_idx] = 1.0

    return legal_mask


def encode_observation(obs: GameObservation):
    """
    This builds the EXACT (127,) state vector for your neural net:
      [ hand_mask(36),
        table_mask(36),
        played_mask(36),
        trump_one_hot(6),
        pos_one_hot(4),
        trick_one_hot(9)
      ]
    plus legal_mask(36)
    """

    # --------------------------
    # 1. Hand mask
    # --------------------------
    hand_int_ids = np.where(obs.hand == 1)[0]   # jass-kit card IDs
    hand_mask = cards_mask_from_int_ids(hand_int_ids)

    # --------------------------
    # 2. Cards on table right now
    # --------------------------
    if obs.nr_played_cards == 36:
        table_mask = np.zeros(NUM_CARDS, dtype=np.float32)
    else:
        trick_row = obs.tricks[obs.nr_tricks]     # int IDs + -1 padding
        table_ids = [cid for cid in trick_row if cid != -1]
        table_mask = cards_mask_from_int_ids(table_ids)

    # --------------------------
    # 3. Played cards mask
    # --------------------------
    played_ids = []
    for t in range(obs.nr_tricks):
        for cid in obs.tricks[t]:
            if cid != -1:
                played_ids.append(cid)
    played_mask = cards_mask_from_int_ids(played_ids)

    # --------------------------
    # 4. Trump one-hot
    # --------------------------
    trump_oh = np.zeros(6, dtype=np.float32)
    if obs.trump in TRUMP_MAP:
        trump_oh[obs.trump] = 1.0

    # --------------------------
    # 5. Player position one-hot
    # --------------------------
    pos_oh = np.zeros(4, dtype=np.float32)
    if obs.player != -1:
        pos_oh[obs.player] = 1.0

    # --------------------------
    # 6. Trick number one-hot
    # --------------------------
    trick_oh = np.zeros(9, dtype=np.float32)
    trick_oh[obs.nr_tricks] = 1.0

    # --------------------------
    # 7. Concatenate
    # --------------------------
    state_vec = np.concatenate([
        hand_mask,
        table_mask,
        played_mask,
        trump_oh,
        pos_oh,
        trick_oh
    ]).astype(np.float32)

    # --------------------------
    # 8. Legal moves
    # --------------------------
    legal_mask = compute_legal_cards(obs)

    return state_vec, legal_mask

@dataclass
class NNState:
    def __init__(self, x: torch.Tensor, legal_mask: torch.Tensor):
        """
        x           : FloatTensor [127]
        legal_mask  : FloatTensor [36]
        """
        self.x = x
        self.legal_mask = legal_mask

    @classmethod
    def from_observation(cls, obs, device="cpu"):
        x_np, legal_np = encode_observation(obs)

        return cls(
            x=torch.from_numpy(x_np).float().to(device),
            legal_mask=torch.from_numpy(legal_np).float().to(device)
        )

    def to_torch(self, device=None):
        """
        Returns batch-shaped tensors for policy net.
        Shapes:
          x:          (1, 127)
          legal_mask: (1, 36)
        """
        if device is None:
            device = self.x.device

        return (
            self.x.unsqueeze(0).to(device),
            self.legal_mask.unsqueeze(0).to(device)
        )