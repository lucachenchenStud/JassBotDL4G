# agent_nn_mcts.py
import torch

from jass.agents.agent import Agent
from jass.game.const import card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber  # adapt if necessary
from nn_state import NNState

from trump_model import TrumpNet          # your existing file
from nn_policy import NNCardPolicy
from mcts_nn import NNMCTS
from jass.agents.agent_mcts_determinized import MCTSAgentDeterminized
import numpy as np
import time


class AgentNNMCTS(Agent):
    """
    Full agent:
      - Trump phase: uses your TrumpPredictor
      - Card phase: NN-guided MCTS
    """

    def __init__(
        self,
        trump_predictor,
        policy: NNCardPolicy,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        gamma: float = 1.0,
        device: torch.device | None = None,
    ):
        self.trump_predictor = trump_predictor
        self.policy = policy
        self.rule = RuleSchieber()
        self.device = device or torch.device("cpu")

        self.mcts = NNMCTS(
            policy=self.policy,
            rule=self.rule,
            n_simulations=n_simulations,
            c_puct=c_puct,
            gamma=gamma,
            device=self.device,
        )

    def action_trump(self, obs: GameObservation) -> int:
        """
        Convert observation → input for TrumpPredictor.
        TrumpPredictor expects a list of card strings + forehand flag.
        """

        # Convert one-hot obs.hand → list of card names
        hand_cards = [card_strings[i] for i in range(36) if obs.hand[i] == 1]

        # Extract forehand: either 0 or 1
        forehand_flag = int(obs.forehand == 1)

        # Call the predictor correctly
        trump_choice = int(self.trump_predictor.predict_trump(hand_cards, forehand_flag))

        mapped_trump = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 10  # important!
        }[trump_choice]

        if mapped_trump == 10 and obs.forehand != -1:
            # PUSH illegal → choose best non-push alternative
            mapped_trump = 4

        return mapped_trump

    def action_play_card(self, obs):
        # 1. Get legal CARDS ONLY
        legal_cards = self.rule.get_valid_cards_from_obs(obs)  # shape (36,)
        legal_ids = np.flatnonzero(legal_cards)

        if len(legal_ids) == 0:
            raise RuntimeError("No legal cards available")

        # 2. NN policy (36-dim)
        state = NNState.from_observation(obs)
        probs = self.policy.policy(state).flatten()

        # 3. Mask illegal
        probs[legal_cards == 0] = -1e9

        # 4. Order legal cards
        ordered_actions = legal_ids[np.argsort(-probs[legal_ids])]

        # 5. Pass ONLY card actions to MCTS
        return self.mcts.select_action(obs, action_order=ordered_actions)
