# mcts_nn.py
import math
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch

from determinize import determinize_state_from_obs
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber  # adapt if your rule class is named differently
from jass.game.const import ACTION_SET_FULL_SIZE

from nn_state import NNState
from nn_policy import NNCardPolicy


@dataclass
class MCTSNode:
    """
    Node for NN-guided MCTS.
    """
    obs: GameObservation                       # determinized observation or state representation
    parent: Optional["MCTSNode"]
    action_from_parent: Optional[int]          # full action index used to reach this node
    prior: float                               # π(a|s) from the NN (for the action that led here)
    value_sum: float = 0.0                     # sum of backed-up values
    visit_count: int = 0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    is_terminal: bool = False
    reward_from_parent: float = 0.0            # immediate reward from parent→this

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def u_value(self, parent_visit_count: int, c_puct: float) -> float:
        if self.visit_count == 0:
            return c_puct * self.prior * math.sqrt(parent_visit_count + 1e-8)
        return c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)


class NNMCTS:
    """
    Neural-network-guided Monte Carlo Tree Search.

    Uses:
      - NNCardPolicy for prior probabilities over legal actions.
      - RuleSchieber to compute legal actions.
      - An environment-specific step function to advance the game given obs + action.
    """

    def __init__(
        self,
        policy: NNCardPolicy,
        rule: RuleSchieber,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        gamma: float = 1.0,
        device: Optional[torch.device] = None,
        seed=0
    ):
        self._action_order = None
        self.rng = np.random.default_rng(seed)
        self.policy = policy
        self.rule = rule
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.device = device or torch.device("cpu")

    # ---------- Public entry point ----------

    def select_action(self, root_obs: GameObservation, action_order=None) -> int:
        self._action_order = action_order

        root_node = self._create_root(root_obs)

        for _ in range(self.n_simulations):
            node, path = self._select(root_node)
            value = self._expand_and_evaluate(node)
            self._backup(path, value)

        best_action, _ = max(
            root_node.children.items(),
            key=lambda kv: kv[1].visit_count
        )
        return int(best_action)

    # ---------- Core MCTS steps ----------

    def _create_root(self, obs: GameObservation) -> MCTSNode:
        """
        Create root node with priors from NN and mark legal actions.
        """
        obs_copy = copy.deepcopy(obs)

        valid_actions = self.rule.get_valid_actions_from_obs(obs_copy)  # shape [ACTION_SET_FULL_SIZE]
        valid_indices = np.nonzero(valid_actions)[0]

        # evaluate NN once for root
        state = NNState.from_observation(obs_copy)
        state_tensor = state.x.unsqueeze(0).to(self.device)           # [1, 127]
        legal_mask = state.legal_mask.unsqueeze(0).to(self.device)    # [1, A]
        with torch.no_grad():
            priors = self.policy.model(state_tensor).softmax(dim=-1)  # [1, A]
            priors = priors * legal_mask
            priors = priors / (priors.sum(dim=-1, keepdim=True) + 1e-8)
            priors = priors.squeeze(0).cpu().numpy()

        root = MCTSNode(
            obs=obs_copy,
            parent=None,
            action_from_parent=None,
            prior=1.0,
            is_terminal=self._is_terminal(obs_copy),
        )

        # children will be expanded lazily, but we keep the prior vector at root
        root.prior_vector = priors
        root.valid_indices = valid_indices
        return root

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Traverse the tree from root using PUCT until we reach a leaf (not fully expanded or terminal).
        Returns:
          leaf_node, path (list of nodes including root and leaf)
        """
        node = root
        path = [node]

        while not node.is_terminal:
            # expand if node has no children yet
            if len(node.children) == 0:
                break

            # select child that maximizes Q + U
            parent_visits = node.visit_count
            best_score = -1e9
            best_child = None

            for action, child in node.children.items():
                q = child.q_value()
                u = child.u_value(parent_visits, self.c_puct)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            node = best_child
            path.append(node)

            if node.is_terminal:
                break

        return node, path

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand leaf node by:
          - computing legal actions
          - creating children with NN priors
          - optionally running a rollout if desired

        Returns:
          value estimate from this node for backup (from current player perspective).
        """
        if node.is_terminal:
            # terminal node: value is just immediate reward (already encoded in node.reward_from_parent)
            # We'll return 0 here and rely on backup to propagate terminal reward only.
            return 0.0

        # 1. get legal actions for this node
        valid_actions = self.rule.get_valid_actions_from_obs(node.obs)
        legal_indices = np.nonzero(valid_actions)[0]

        if len(legal_indices) == 0:
            node.is_terminal = True
            return 0.0

        # 2. get NN priors
        state = NNState.from_observation(node.obs)
        state_tensor = state.x.unsqueeze(0).to(self.device)
        legal_mask = state.legal_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy.model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            probs = probs * legal_mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            priors = probs.squeeze(0).cpu().numpy()

        # 3. create child nodes for *all* legal actions
        for a in legal_indices:
            if a in node.children:
                continue

            prior_a = float(priors[a])

            # ----- ENV-SPECIFIC STEP 1: apply action to get next obs + reward -----
            next_obs, reward, terminal = self._step_env(node.obs, int(a))
            # ---------------------------------------------------------------------

            child = MCTSNode(
                obs=next_obs,
                parent=node,
                action_from_parent=int(a),
                prior=prior_a,
                is_terminal=terminal,
                reward_from_parent=reward,
            )
            node.children[int(a)] = child

        # 4. leaf node value estimate from NN (optional: rollout instead)
        value = self._nn_value(node.obs)

        return float(value)

    def _backup(self, path: List[MCTSNode], leaf_value: float) -> None:
        """
        Backup value estimates and rewards along the path.
        Assumes leaf_value is from the perspective of the player at the *last* node.
        Here we do a simple "same player perspective" backup:
          G_t = r_{t+1} + γ r_{t+2} + ... + γ^{k-1} r_{t+k}
        and we add leaf_value at the leaf.
        """
        # accumulate discounted future rewards including leaf_value
        returns = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += returns
            returns = node.reward_from_parent + self.gamma * returns

    # ---------- NN value estimate ----------

    def _nn_value(self, obs: GameObservation) -> float:
        """
        Simple NN-based value estimate: we reuse the policy net,
        but you could plug in a separate value head here.
        For now we just return 0.0 (pure policy-guided MCTS).
        """
        # If you later add a value head, implement it here.
        return 0.0

    # ---------- Terminal & environment step stubs ----------

    def _is_terminal(self, obs: GameObservation) -> bool:
        """
        Terminal if 36 cards were played (9 tricks * 4 cards).
        """
        return obs.nr_played_cards >= 36

    def _step_env(self, obs, action: int):
        """
        Advance the game by ONE action.
        """

        # 1. Determinize hidden information
        state = determinize_state_from_obs(obs, self.rng)

        # 2. Init simulator
        sim = GameSim(self.rule)
        sim.init_from_state(state)

        # 3. Apply action
        if state.trump == -1:
            sim.action_trump(int(action))
        else:
            sim.action_play_card(int(action))

        # 4. Read next state
        next_state = sim.state
        data = next_state.to_json()

        # Set correct player view
        data["playerView"] = next_state.player

        for i in range(4):
            if i != next_state.player:
                data["player"][i]["hand"] = []

        next_obs = GameObservation.from_json(data)

        # 5. Terminal?
        terminal = (next_state.nr_played_cards == 36)

        # 6. Reward (simple and valid)
        reward = 0.0
        if terminal:
            team = obs.player_view % 2
            reward = float(
                next_state.points[team]
                - next_state.points[1 - team]
            )

        return next_obs, reward, terminal
