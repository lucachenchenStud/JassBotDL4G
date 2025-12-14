"""
agent_mcts_determinized.py
--------------------------
Determinized Monte Carlo Tree Search (MCTS) agent for Jass.

Compatible with HSLU / DL4G Jass environment (jass.game.*).
"""

import numpy as np
import math
import copy
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.game.const import *


# ---------------------------------------------------------------------------
# Local helper functions
# ---------------------------------------------------------------------------

def get_valid_moves(rule, game):
    """Return all valid card indices for the current player."""
    valid = rule.get_valid_cards_from_state(game.state)
    return np.flatnonzero(valid)


def team_of(player_id: int) -> int:
    """Return the team ID (0 or 1) for a given player."""
    return 0 if player_id in [0, 2] else 1


def value_from_points(points: np.ndarray, team_id: int) -> float:
    """Compute normalized score difference for our team."""
    team_points = points[team_id]
    opp_points = points[1 - team_id]
    return (team_points - opp_points) / 157.0  # max points in Schieber Jass


# ---------------------------------------------------------------------------
# Determinization wrapper
# ---------------------------------------------------------------------------

class MCTSAgentDeterminized:
    """
    Determinized MCTS agent that samples hidden hands consistent with the current observation,
    runs MCTS on each determinization, and aggregates visit counts.
    """

    def __init__(self, iterations_per_det=300, n_dets=20, c=1.2, horizon_tricks=1):
        self.iterations_per_det = iterations_per_det
        self.n_dets = n_dets
        self.c = c
        self.horizon_tricks = horizon_tricks

    def action_play_card(self, obs):
        visit_counts = {}
        for _ in range(self.n_dets):
            g_det = sample_determinization(obs)
            mcts = MCTSAgentCheating(
                rule=RuleSchieber(),
                iterations=self.iterations_per_det,
                c=self.c,
                horizon_tricks=self.horizon_tricks
            )
            a = mcts.select_card(g_det)
            visit_counts[a] = visit_counts.get(a, 0) + 1

        best_action = max(visit_counts.items(), key=lambda kv: kv[1])[0]
        return best_action


# ---------------------------------------------------------------------------
# Determinization sampling
# ---------------------------------------------------------------------------

def sample_determinization(obs, rng=np.random):
    """Return a full GameSim (complete information) consistent with the observation."""
    my_id = obs.player
    my_hand = np.flatnonzero(obs.hand)

    # --- 1. Collect played cards ---
    played = set()
    for trick in obs.tricks:
        for c in trick:
            if c != -1:
                played.add(int(c))
    for c in obs.current_trick:
        if c != -1:
            played.add(int(c))

    all_cards = set(range(36))
    remaining = list(all_cards - set(my_hand) - played)
    rng.shuffle(remaining)

    # --- 2. Count how many cards each player has played ---
    cards_played_by = [0, 0, 0, 0]
    current_first_player = int(obs.trick_first_player[obs.nr_tricks])

    for t_index, t in enumerate(obs.tricks):
        first_p = int(obs.trick_first_player[t_index])
        for i, c in enumerate(t):
            if c != -1:
                pid = (first_p + i) % 4
                cards_played_by[pid] += 1

    for i, c in enumerate(obs.current_trick):
        if c != -1:
            pid = (current_first_player + i) % 4
            cards_played_by[pid] += 1

    # --- 3. Determine remaining cards per player ---
    need = [9 - cards_played_by[p] for p in range(4)]
    need[my_id] = len(my_hand)

    # --- 4. Assign unknown cards randomly to opponents ---
    cursor = 0
    hands = np.zeros((4, 36), dtype=np.int8)
    hands[my_id, my_hand] = 1
    opp_ids = [p for p in range(4) if p != my_id]

    for pid in opp_ids:
        k = need[pid]
        assign = remaining[cursor:cursor + k]
        cursor += k
        hands[pid, assign] = 1

    # --- 5. Build a GameSim with copied state ---
    g = GameSim(rule=RuleSchieber())
    s = g.state
    s.hands = hands
    s.player = obs.player
    s.trump = obs.trump
    s.points = np.array(obs.points, dtype=np.int16)
    s.dealer = obs.dealer
    s.nr_tricks = obs.nr_tricks
    s.nr_played_cards = obs.nr_played_cards
    s.nr_cards_in_trick = obs.nr_cards_in_trick
    s.trick_first_player = np.array(obs.trick_first_player, dtype=np.int16)
    s.trick_points = np.array(obs.trick_points, dtype=np.int16)
    s.trick_winner = np.array(obs.trick_winner, dtype=np.int16)
    s.current_trick = np.array(obs.current_trick, dtype=np.int16)
    s.tricks = np.array(obs.tricks, dtype=np.int16)
    s.forehand = obs.forehand
    s.declared_trump = obs.declared_trump

    return g


# ---------------------------------------------------------------------------
# Full-information MCTS core
# ---------------------------------------------------------------------------

class MCTSAgentCheating:
    """Full-information MCTS used on determinized states."""

    def __init__(self, rule, iterations=1000, c=1.414, horizon_tricks=1):
        self.rule = rule
        self.iterations = iterations
        self.c = c
        self.horizon_tricks = horizon_tricks

    def select_card(self, game):
        root_team = team_of(game.state.player)
        root = MCTSNode(untried=get_valid_moves(self.rule, game))

        if len(root.untried) == 1:
            return root.untried[0]

        for _ in range(self.iterations):
            node = root
            g = copy.deepcopy(game)

            # --- Selection ---
            while not node.untried and node.children:
                action, node = node.uct_select(self.c)
                g.action_play_card(action)

            # --- Expansion ---
            if node.untried:
                a = np.random.choice(node.untried)
                g.action_play_card(a)
                child_untried = get_valid_moves(self.rule, g)
                node = node.expand(a, child_untried)

            # --- Simulation ---
            simulate_random_until(self.rule, g, n_tricks=self.horizon_tricks)

            # --- Evaluation ---
            val = value_from_points(g.state.points, root_team)

            # --- Backpropagation ---
            node.backprop(val)

        best_a = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_a


# ---------------------------------------------------------------------------
# Random rollout helper
# ---------------------------------------------------------------------------

def simulate_random_until(rule, game, n_tricks=1):
    """Simulate a few random tricks for fast rollout."""
    for _ in range(n_tricks * 4):
        if game.state.nr_tricks >= 9:
            break
        valid = get_valid_moves(rule, game)
        if not len(valid):
            break
        a = np.random.choice(valid)
        game.action_play_card(a)


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("parent", "children", "N", "W", "Q", "untried")

    def __init__(self, parent=None, untried=None):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.untried = list(untried) if untried is not None else []

    def uct_select(self, c=1.414):
        log_Np = math.log(self.N + 1e-9)

        def uct(a_node):
            a, node = a_node
            return node.Q + c * math.sqrt(log_Np / (node.N + 1e-9))

        return max(self.children.items(), key=uct)

    def expand(self, action, child_untried):
        child = MCTSNode(parent=self, untried=child_untried)
        self.children[action] = child
        if action in self.untried:
            self.untried.remove(action)
        return child

    def backprop(self, value):
        n = self
        while n is not None:
            n.N += 1
            n.W += value
            n.Q = n.W / n.N
            n = n.parent
