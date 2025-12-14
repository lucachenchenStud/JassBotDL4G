# state_encoder.py

import numpy as np

from cards import CARD_TO_INDEX, CARD_LIST
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

_rule = RuleSchieber()

# Trump mapping
TRUMP_MAP = {
    "DIAMONDS": 0,
    "HEARTS": 1,
    "SPADES": 2,
    "CLUBS": 3,
    "OBE_ABE": 4,
    "UNE_UFE": 5
}


def encode_cards_mask(card_list):
    """Return a 36-dim mask for a list of card strings."""
    mask = np.zeros(36, dtype=np.float32)
    for c in card_list:
        mask[CARD_TO_INDEX[c]] = 1.0
    return mask


def encode_trump(trump):
    """Accepts both int (0..5) and string trump names."""
    oh = np.zeros(6, dtype=np.float32)

    # If trump is already an integer
    if isinstance(trump, int):
        if 0 <= trump < 6:
            oh[trump] = 1.0
            return oh
        else:
            raise ValueError(f"Invalid trump integer: {trump}")

    # If trump is a string -> use TRUMP_MAP
    if isinstance(trump, str):
        oh[TRUMP_MAP[trump]] = 1.0
        return oh

    raise TypeError(f"Unsupported trump type: {type(trump)}")



def encode_player_pos(player_pos):
    """Encode player position 0–3 into one-hot."""
    oh = np.zeros(4, dtype=np.float32)
    oh[player_pos] = 1.0
    return oh


def encode_trick_number(trick_no):
    """Encode trick number (0–8) into one-hot."""
    oh = np.zeros(9, dtype=np.float32)
    oh[trick_no] = 1.0
    return oh


def encode_sample(sample):
    """
    Adapted to the sample format you provided.
    Expected keys:
      'hand'
      'table'
      'played_cards'
      'trump'
      'player'          -> player_pos
      'trick'           -> trick_no
      'valid_moves'     -> legal_moves
      'action'
    """

    # 1. Encode masks
    hand_mask   = encode_cards_mask(sample["hand"])
    table_mask  = encode_cards_mask(sample["table"])
    played_mask = encode_cards_mask(sample["played_cards"])

    # 2. Meta information
    trump_oh = encode_trump(sample["trump"])
    pos_oh   = encode_player_pos(sample["player"])
    trick_oh = encode_trick_number(sample["trick"])

    # 3. Legal moves
    legal_mask = encode_cards_mask(sample["valid_moves"])

    # 4. Target action
    action_index = CARD_TO_INDEX[sample["action"]]

    # 5. Build input vector
    state_vector = np.concatenate([
        hand_mask,
        table_mask,
        played_mask,
        trump_oh,
        pos_oh,
        trick_oh
    ]).astype(np.float32)

    return state_vector, legal_mask.astype(np.float32), action_index


def encode_observation(obs: GameObservation):
    """
    Encode a GameObservation into NN inputs.
    Returns:
      state_vector: float32 [127]
      legal_mask:   float32 [36]
    """

    # --- Cards ---
    hand_mask = obs.hand.astype(np.float32)

    # current trick cards
    table_cards = []
    if obs.current_trick is not None:
        for c in obs.current_trick:
            if c != -1:
                table_cards.append(c)

    table_mask = np.zeros(36, dtype=np.float32)
    for c in table_cards:
        table_mask[c] = 1.0

    # played cards = all cards in previous tricks
    played_mask = np.zeros(36, dtype=np.float32)
    for trick in obs.tricks:
        for c in trick:
            if c != -1:
                played_mask[c] = 1.0

    # --- Meta ---
    trump_oh = encode_trump(obs.trump)
    pos_oh   = encode_player_pos(obs.player_view)
    trick_oh = encode_trick_number(obs.nr_tricks)

    # --- Legal moves ---
    legal_mask = _rule.get_valid_cards_from_obs(obs).astype(np.float32)

    # --- Final vector ---
    state_vector = np.concatenate([
        hand_mask,
        table_mask,
        played_mask,
        trump_oh,
        pos_oh,
        trick_oh
    ]).astype(np.float32)

    return state_vector, legal_mask