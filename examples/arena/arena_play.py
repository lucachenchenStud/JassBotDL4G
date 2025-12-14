# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import numpy as np
import torch
import os

from agent_nn_mcts import AgentNNMCTS
from agent_nn_only import AgentNNOnly
from card_model import CardPolicyNetwork
from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import color_masks, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from nn_policy import NNCardPolicy
from trump_model import TrumpNet, TrumpMLP


class MyAgent(Agent):
    """
    Sample implementation of a player to play Jass.
    """
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (obs.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        # cards are one hot encoded

        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        card = self._rng.choice(np.flatnonzero(valid_cards))
        self._logger.debug('Played card: {}'.format(card_strings[card]))
        return card


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # directory of the current file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # path to the models folder
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # setup the arena
    arena = Arena(nr_games_to_play=100, save_filename='arena_games')
    player = AgentRandomSchieber()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trump_model = TrumpMLP(model_path=os.path.join(MODEL_DIR,"trump_model.pt"), device=device)
    nn_card_policy = NNCardPolicy(model_class=CardPolicyNetwork, model_path=os.path.join(MODEL_DIR,"card_model.pt"), device = device)
    my_player = AgentNNOnly(trump_model, nn_card_policy)
    mcts_player = AgentNNMCTS(
        trump_predictor=trump_model,
        policy=nn_card_policy,
        n_simulations=3,
        c_puct=1.5,
        gamma=1.0,
        device=device,
    )

    arena.set_players(my_player, player, my_player, player)
    #arena.set_players(my_player, mcts_player, my_player, mcts_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))
    print('Points Team 0: {:.2f})'.format(arena.points_team_0.sum()))
    print('Points Team 1: {:.2f})'.format(arena.points_team_1.sum()))

if __name__ == '__main__':
    main()
