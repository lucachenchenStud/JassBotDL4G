# agent_nn_only.py

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

from nn_state import NNState               # must be implemented
from nn_policy import NNCardPolicy         # must be implemented


class AgentNNOnly(Agent):
    """
    A fully neural Jass agent with:
      - a trump predictor (predict_trump(obs) -> int)
      - a neural card policy (NNCardPolicy)

    This agent already conforms to the Jass-kit Agent interface,
    so it works directly with the Arena.
    """

    def __init__(self, trump_predictor, policy: NNCardPolicy):
        super().__init__()
        self.trump_predictor = trump_predictor
        self.policy = policy
        self.rule = RuleSchieber()  # ensures we always obey Jass rules

    # --------------------------------------------------------------------------
    # Trump selection
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Card playing
    # --------------------------------------------------------------------------
    def action_play_card(self, obs: GameObservation) -> int:
        """
        Called by Arena when a card must be played.
        Must return an integer card ID [0–35].
        """

        if obs.player != obs.player_view:
            raise RuntimeError(
                f"Agent called when not its turn: "
                f"player={obs.player}, view={obs.player_view}"
            )

        # Compute legal actions FIRST
        legal_mask = self.rule.get_valid_cards_from_obs(obs)
        legal_indices = np.flatnonzero(legal_mask)

        if len(legal_indices) == 0:
            raise RuntimeError(
                "No legal cards available.\n"
                f"player={obs.player}\n"
                f"hand={obs.hand}\n"
                f"tricks={obs.tricks}\n"
                f"current_trick={obs.current_trick}\n"
                f"nr_tricks={obs.nr_tricks}\n"
            )

        # Convert observation → neural network input
        state = NNState.from_observation(obs)

        # NN proposal
        card_id = int(self.policy.greedy_action(state))

        # Repair if illegal
        if legal_mask[card_id] != 1:
            card_id = int(np.random.choice(legal_indices))

        return card_id

    def onehot_to_cards(self, onehot):
        """Convert a 36-dim one-hot hand array into list of card strings."""
        return [card_strings[i] for i in range(36) if onehot[i] == 1]