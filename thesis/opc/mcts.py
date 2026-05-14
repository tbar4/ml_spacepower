"""MCTS implementation for OPC AlphaZero.

Each search is rooted at the current game state. The neural network provides
priors and value estimates. Both players share one network; each player feeds
their own observation tensor to get their perspective's value and policy.

For imperfect-information OPC: MCTS expands true states (it sees the full
simulator), but the neural net is evaluated on the player's observation tensor
(noisy for the defender). This is the standard approximation used in
neural-net-guided MCTS on partially observable games.
"""

import math
import numpy as np
import torch

from opc.game import _N_JOINT, _OBS_SIZE, ATTACKER_ID, DEFENDER_ID


# ── MCTS configuration ────────────────────────────────────────────────────────

C_PUCT = 1.25       # UCB exploration constant
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMPERATURE_MOVES = 20  # use full temperature sampling for first N moves per episode


# ── Tree node ─────────────────────────────────────────────────────────────────

class Node:
    """One node in the MCTS tree, corresponding to a game state."""

    __slots__ = (
        "state", "parent", "action_from_parent",
        "children", "prior",
        "visit_count", "total_value",
        "is_expanded",
    )

    def __init__(self, state, parent=None, action_from_parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior
        self.children: dict[int, "Node"] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits: int) -> float:
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self) -> tuple[int, "Node"]:
        """Select child with highest UCB score."""
        total_n = self.visit_count
        best_score = -float("inf")
        best_action = -1
        best_child = None
        for action, child in self.children.items():
            score = child.ucb_score(total_n)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child


def _obs_tensor(state, player: int) -> torch.Tensor:
    """Return observation tensor for player as a float32 tensor."""
    obs = state.observation_tensor(player)
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


# ── MCTS search ───────────────────────────────────────────────────────────────

class MCTS:
    """AlphaZero-style MCTS for OPC."""

    def __init__(self, network: torch.nn.Module, n_simulations: int = 200):
        self.network = network
        self.n_simulations = n_simulations

    def run(
        self,
        root_state,
        add_dirichlet_noise: bool = True,
    ) -> dict[int, float]:
        """Run MCTS from root_state and return action visit-count distribution."""
        root = Node(state=root_state.clone())
        root = self._expand(root)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_expanded and not node.state.is_terminal():
                _, node = node.select_child()
                search_path.append(node)

            # Expansion and evaluation
            if not node.state.is_terminal():
                node = self._expand(node)
                value = self._evaluate(node)
            else:
                # Terminal: use actual return from current player's perspective
                returns = node.state.returns()
                player = search_path[-2].state.current_player() if len(search_path) > 1 else ATTACKER_ID
                value = returns[player]

            # Backpropagation
            self._backprop(search_path, value)

        return {a: child.visit_count for a, child in root.children.items()}

    def _expand(self, node: Node) -> Node:
        """Expand node: compute priors via network and create child nodes."""
        player = node.state.current_player()
        obs = _obs_tensor(node.state, player)

        with torch.no_grad():
            logits, _ = self.network(obs)
            priors = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        for action in node.state.legal_actions():
            child_state = node.state.clone()
            child_state.apply_action(action)
            child = Node(
                state=child_state,
                parent=node,
                action_from_parent=action,
                prior=float(priors[action]),
            )
            node.children[action] = child

        node.is_expanded = True
        return node

    def _evaluate(self, node: Node) -> float:
        """Evaluate a leaf node using the value network.

        The network is trained to predict the acting player's expected return
        given their observation. We negate to convert to the parent's perspective,
        which is what _backprop stores in each node's total_value.
        """
        player = node.state.current_player()
        obs = _obs_tensor(node.state, player)
        with torch.no_grad():
            _, value = self.network(obs)
        return -float(value.squeeze())

    def _backprop(self, search_path: list[Node], leaf_value: float):
        """Backpropagate value up the search path, negating at player switches."""
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            # Negate when crossing a player boundary
            if node.parent is not None:
                parent_player = node.parent.state.current_player()
                node_player = node.state.current_player() if not node.state.is_terminal() else -1
                if parent_player != node_player:
                    value = -value

    @staticmethod
    def _add_dirichlet_noise(root: Node):
        actions = list(root.children.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * n

    @staticmethod
    def visit_counts_to_policy(
        visit_counts: dict[int, float],
        n_actions: int = _N_JOINT,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Convert visit counts to a policy vector (with temperature)."""
        counts = np.zeros(n_actions, dtype=np.float32)
        for a, n in visit_counts.items():
            counts[a] = n

        if temperature < 1e-6:
            # Greedy
            policy = np.zeros_like(counts)
            policy[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            total = counts.sum()
            policy = counts / total if total > 0 else np.ones(n_actions) / n_actions
        return policy
