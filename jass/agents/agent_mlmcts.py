import numpy as np
import pandas as pd
import joblib
from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber

# -------------------------------------------------------------------
# Mapping and constants
# -------------------------------------------------------------------
LABEL2TRUMP = {
    "DIAMONDS": DIAMONDS,
    "HEARTS": HEARTS,
    "SPADES": SPADES,
    "CLUBS": CLUBS,
    "OBE_ABE": OBE_ABE,
    "UNE_UFE": UNE_UFE,
    "PUSH": PUSH,
}

CARDS = [
    "DA","DK","DQ","DJ","D10","D9","D8","D7","D6",
    "HA","HK","HQ","HJ","H10","H9","H8","H7","H6",
    "SA","SK","SQ","SJ","S10","S9","S8","S7","S6",
    "CA","CK","CQ","CJ","C10","C9","C8","C7","C6"
]


# -------------------------------------------------------------------
# Trump predictor (ML model)
# -------------------------------------------------------------------
class TrumpPredictor:
    def __init__(self, model_path: str = "trump_model.joblib"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]

    def _build_features(self, hand_cards_bool: dict, is_forehand: bool) -> pd.DataFrame:
        row = {c: bool(hand_cards_bool.get(c, False)) for c in self.feature_columns if c != "FH"}
        row["FH"] = bool(is_forehand)
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, hand_cards_bool: dict, is_forehand: bool) -> str:
        X = self._build_features(hand_cards_bool, is_forehand)
        return self.model.predict(X)[0]

    def predict_proba(self, hand_cards_bool: dict, is_forehand: bool):
        X = self._build_features(hand_cards_bool, is_forehand)
        probs = self.model.predict_proba(X)[0]
        return dict(zip(self.model.classes_, probs))

    def prepare_input(self, obs):
        features = np.concatenate([obs.hand.astype(int), [int(obs.forehand)]])
        return features.reshape(1, -1)

    def label_to_trump(self, label):
        return LABEL2TRUMP[label]


# -------------------------------------------------------------------
# MCTS adapter to unify interfaces
# -------------------------------------------------------------------
class MCTSAdapter:
    """Wrapper to unify MCTSAgentDeterminized interface with Agent.play_card."""
    def __init__(self, mcts_agent):
        self.mcts_agent = mcts_agent

    def select_card(self, obs):
        return self.mcts_agent.action_play_card(obs)


# -------------------------------------------------------------------
# Combined Agent (ML + MCTS)
# -------------------------------------------------------------------
class AgentMLMCTS(Agent):
    """
    Jass Agent combining:
      - ML model for trump prediction
      - Determinized MCTS for card play
    """
    def __init__(self, trump_predictor: TrumpPredictor, mcts_adapter: MCTSAdapter):
        super().__init__()
        self._rule = RuleSchieber()
        self.trump_predictor = trump_predictor
        self.mcts_adapter = mcts_adapter

    def action_trump(self, obs, json_payload=None) -> int:
        """
        Predict trump using the trained ML model.
        During trump selection, the cards are provided as strings in the JSON, not in obs.hand.
        """
        # 1) Extract hand booleans
        hand_bool = {c: False for c in CARDS}

        # Prefer JSON hand (strings) if provided and trump not set yet
        if json_payload is not None and obs.trump == -1:
            pidx = json_payload.get("playerView", json_payload.get("currentPlayer", getattr(obs, "player", 0)))
            cards_list = []
            try:
                cards_list = json_payload["player"][int(pidx)].get("hand", []) or []
            except Exception:
                cards_list = []
            for s in cards_list:
                if s in hand_bool:
                    hand_bool[s] = True
        else:
            # Fallback: one-hot from obs.hand if it exists
            try:
                for i, c in enumerate(CARDS):
                    hand_bool[c] = bool(obs.hand[i])
            except Exception:
                pass

        # Optional interaction features if your model expects them
        for color in "DHSC":
            j9 = hand_bool.get(f"{color}J", False) and hand_bool.get(f"{color}9", False)
            akq = hand_bool.get(f"{color}A", False) and hand_bool.get(f"{color}K", False) and hand_bool.get(f"{color}Q", False)
            if f"{color}_J9" in self.trump_predictor.feature_columns:
                hand_bool[f"{color}_J9"] = j9
            if f"{color}_AKQ" in self.trump_predictor.feature_columns:
                hand_bool[f"{color}_AKQ"] = akq

        # 2) Predict label then map to enum
        label = self.trump_predictor.predict(hand_bool, bool(getattr(obs, "forehand", 0)))

        # Optional: avoid PUSH if not confident
        if label == "PUSH":
            proba = self.trump_predictor.predict_proba(hand_bool, bool(getattr(obs, "forehand", 0)))
            best_non_push = max({k: v for k, v in proba.items() if k != "PUSH"}, key=lambda k: proba[k])
            if proba[best_non_push] > 0.35:
                label = best_non_push

        return LABEL2TRUMP[label]

    def action_play_card(self, obs, json_payload=None):
        """
        Choose card to play using the MCTS adapter.
        Ensures obs.hand is usable; robust fallback uses rule-valid cards.
        """
        # If obs.hand may be empty in your environment, ensure it from JSON (like in app.py)
        try:
            if int(np.sum(obs.hand)) == 0 and json_payload is not None:
                CARDS = ["DA", "DK", "DQ", "DJ", "D10", "D9", "D8", "D7", "D6",
                         "HA", "HK", "HQ", "HJ", "H10", "H9", "H8", "H7", "H6",
                         "SA", "SK", "SQ", "SJ", "S10", "S9", "S8", "S7", "S6",
                         "CA", "CK", "CQ", "CJ", "C10", "C9", "C8", "C7", "C6"]
                idx = {c: i for i, c in enumerate(CARDS)}
                pidx = json_payload.get("playerView", json_payload.get("currentPlayer", getattr(obs, "player", 0)))
                cards_str = json_payload["player"][int(pidx)].get("hand", []) or []
                vec = np.zeros(36, dtype=np.int8)
                for s in cards_str:
                    if s in idx:
                        vec[idx[s]] = 1
                obs.hand = vec
        except Exception:
            pass

        # Try MCTS; if it fails or returns nothing, fall back to rule-valid random
        try:
            return self.mcts_adapter.select_card(obs)
        except Exception as e:
            print(f"[WARN] MCTS failed ({e}), playing random valid card.")
            valid_mask = self._rule.get_valid_cards_from_obs(obs)
            valid = np.flatnonzero(valid_mask)
            if len(valid) == 0:
                raise RuntimeError("No valid moves available in fallback")
            return int(np.random.choice(valid))