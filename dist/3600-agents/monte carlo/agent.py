from collections.abc import Callable
from typing import List, Tuple, Optional
import random
import math
import time
import numpy as np

from game import board as board_module, move as move_module, enums
from game.enums import (
    MoveType, Direction, Cell, Noise,
    BOARD_SIZE, CARPET_POINTS_TABLE, RAT_BONUS, RAT_PENALTY
)
from game.move import Move


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

UCB_C = 1.4          # exploration constant for UCB1
SEARCH_EV_THRESHOLD = 2.5   # min expected value before we consider a rat search
MIN_BELIEF_TO_SEARCH = 0.55 # only search if top cell belief exceeds this

# Noise emission probabilities  P(noise | cell_type)
# Rows: BLOCKED=3, SPACE=0, PRIMED=1, CARPET=2  (Cell enum values)
NOISE_EMIT = {
    Cell.BLOCKED: np.array([0.5,  0.3,  0.2]),   # squeak scratch squeal
    Cell.SPACE:   np.array([0.7,  0.15, 0.15]),
    Cell.PRIMED:  np.array([0.1,  0.8,  0.1]),
    Cell.CARPET:  np.array([0.1,  0.1,  0.8]),
}

# Distance error offsets and their probabilities
DIST_OFFSETS = np.array([-1, 0, 1, 2])
DIST_PROBS   = np.array([0.12, 0.70, 0.12, 0.06])


# ─────────────────────────────────────────────
#  HMM Rat Tracker
# ─────────────────────────────────────────────

class RatTracker:
    """
    Maintains a probability distribution (belief) over all 64 cells
    for where the rat currently is, updated each turn using:
      - Prediction:  belief = T^T @ belief   (rat transition step)
      - Update:      belief *= P(noise|cell) * P(dist_obs|true_dist)
    """

    def __init__(self, T: np.ndarray):
        # T[i, j] = P(rat moves from cell i to cell j)
        # For prediction we want: new_belief[j] = sum_i belief[i] * T[i,j]
        # which is equivalent to T.T @ belief
        self.T = T                              # shape (64, 64)
        self.belief = np.ones(64) / 64.0        # uniform prior

        # Precompute cell-type arrays (will be updated each turn from board)
        self._cell_type_cache = None

    def predict(self):
        """Propagate belief through transition matrix (rat moves one step)."""
        self.belief = self.T.T @ self.belief
        # Renormalise to correct floating point drift
        s = self.belief.sum()
        if s > 0:
            self.belief /= s

    def update(self, board, noise: int, dist_obs: int, worker_pos: Tuple[int, int]):
        """
        Reweight belief using the observed noise and distance.

        Parameters:
            board: current Board object (used to get cell types)
            noise: int — Noise enum value (0=squeak, 1=scratch, 2=squeal)
            dist_obs: int — observed (noisy) Manhattan distance
            worker_pos: (x, y) of our worker
        """
        wx, wy = worker_pos
        for idx in range(64):
            x = idx % BOARD_SIZE
            y = idx // BOARD_SIZE

            # --- Noise likelihood ---
            cell = board.get_cell((x, y))
            noise_probs = NOISE_EMIT.get(cell, NOISE_EMIT[Cell.SPACE])
            p_noise = noise_probs[noise]

            # --- Distance likelihood ---
            true_dist = abs(x - wx) + abs(y - wy)
            # dist_obs = true_dist + offset, so offset = dist_obs - true_dist
            p_dist = 0.0
            for offset, prob in zip(DIST_OFFSETS, DIST_PROBS):
                # Clamp: game never returns dist < 0
                observed = max(0, true_dist + offset)
                if observed == dist_obs:
                    p_dist += prob

            self.belief[idx] *= p_noise * p_dist

        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            # Degenerate: reset to uniform so we don't get stuck
            self.belief = np.ones(64) / 64.0

    def top_cell(self) -> Tuple[Tuple[int, int], float]:
        """Return ((x,y), probability) of the most likely rat location."""
        idx = int(np.argmax(self.belief))
        prob = float(self.belief[idx])
        x, y = idx % BOARD_SIZE, idx // BOARD_SIZE
        return (x, y), prob

    def search_expected_value(self) -> float:
        """
        Expected value of searching the top cell:
          EV = prob_top * RAT_BONUS - (1 - prob_top) * RAT_PENALTY
        """
        _, prob = self.top_cell()
        return prob * RAT_BONUS - (1 - prob) * RAT_PENALTY

    def mark_caught(self):
        """Reset belief to uniform after catching the rat (new rat spawned)."""
        self.belief = np.ones(64) / 64.0

    def mark_missed(self, loc: Tuple[int, int]):
        """Zero out the cell we just searched and renormalise."""
        idx = loc[1] * BOARD_SIZE + loc[0]
        self.belief[idx] = 0.0
        s = self.belief.sum()
        if s > 0:
            self.belief /= s


# ─────────────────────────────────────────────
#  Board Heuristic
# ─────────────────────────────────────────────

def heuristic(board, rat_belief: np.ndarray) -> float:
    """
    Estimate the value of a board state from the current player's perspective.

    Components:
      1. Score differential
      2. Primed squares we own (potential future carpet rolls)
      3. Longest available carpet roll we could make right now
      4. Expected value of the rat (probabilistic bonus)
    """
    score_diff = board.player_worker.get_points() - board.opponent_worker.get_points()

    # Count primed squares reachable from our position
    # (proxy for carpet potential — just count total primed on board for simplicity)
    primed_count = bin(board._primed_mask).count('1')

    # Longest carpet move currently available
    best_carpet = 0
    for m in board.get_valid_moves(exclude_search=True):
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
            if pts > best_carpet:
                best_carpet = pts

    # Rat expected value contribution
    # Weight by belief: if we're about to search, factor in expected gain
    top_idx = int(np.argmax(rat_belief))
    top_prob = float(rat_belief[top_idx])
    rat_ev = top_prob * RAT_BONUS - (1 - top_prob) * RAT_PENALTY

    return (
        score_diff * 3.0
        + primed_count * 0.3
        + best_carpet * 0.5
        + rat_ev * 0.8
    )


# ─────────────────────────────────────────────
#  MCTS Node
# ─────────────────────────────────────────────

class MCTSNode:
    __slots__ = (
        'board', 'move', 'parent',
        'children', 'untried_moves',
        'wins', 'visits', 'is_maximising'
    )

    def __init__(self, board, move: Optional[Move], parent: Optional['MCTSNode'], is_maximising: bool):
        self.board = board
        self.move = move                  # move that led to this node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.untried_moves: Optional[List[Move]] = None  # lazily initialised
        self.wins = 0.0
        self.visits = 0
        self.is_maximising = is_maximising

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def ucb1(self, c: float = UCB_C) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = UCB_C) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb1(c))

    def best_move_child(self) -> 'MCTSNode':
        """Pick child with most visits (most robust choice at root)."""
        return max(self.children, key=lambda n: n.visits)


# ─────────────────────────────────────────────
#  MCTS Engine
# ─────────────────────────────────────────────

class MCTS:
    def __init__(self, rat_tracker: RatTracker, rollout_depth: int = 8):
        self.rat_tracker = rat_tracker
        self.rollout_depth = rollout_depth

    def _get_moves(self, board) -> List[Move]:
        """Get candidate moves, excluding search (handled separately)."""
        return board.get_valid_moves(exclude_search=True)

    def _rollout_policy(self, moves: List[Move]) -> Move:
        """
        Biased random rollout: prefer carpet > prime > plain.
        This is a lightweight playout policy that steers rollouts
        toward higher-value actions without expensive lookahead.
        """
        carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET]
        prime_moves  = [m for m in moves if m.move_type == MoveType.PRIME]

        if carpet_moves:
            # Among carpet moves, prefer longer rolls
            best_len = max(m.roll_length for m in carpet_moves)
            best_carpets = [m for m in carpet_moves if m.roll_length == best_len]
            return random.choice(best_carpets)
        if prime_moves and random.random() < 0.7:
            return random.choice(prime_moves)
        return random.choice(moves)

    def _rollout(self, board) -> float:
        """
        Simulate to a fixed depth using the rollout policy.
        Returns a heuristic score from the root player's perspective.
        """
        sim_board = board.get_copy()
        # Track whose perspective we started from
        # After each move, we reverse perspective, so 'player' alternates.
        # We want final score from the ORIGINAL player's viewpoint.
        depth = 0
        player_turn = True  # True = original player's turn

        while depth < self.rollout_depth and not sim_board.is_game_over():
            moves = self._get_moves(sim_board)
            if not moves:
                break
            m = self._rollout_policy(moves)
            sim_board.apply_move(m, check_ok=False)
            sim_board.reverse_perspective()
            player_turn = not player_turn
            depth += 1

        # Score from the perspective of whoever holds player_worker NOW.
        # If player_turn is True, player_worker IS the original player.
        score = heuristic(sim_board, self.rat_tracker.belief)
        if not player_turn:
            score = -score
        return score

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add one untried child node."""
        if node.untried_moves is None:
            node.untried_moves = self._get_moves(node.board)
            random.shuffle(node.untried_moves)

        m = node.untried_moves.pop()
        child_board = node.board.forecast_move(m, check_ok=False)
        if child_board is None:
            # Shouldn't happen, but be safe
            return node
        child_board.reverse_perspective()
        child = MCTSNode(
            board=child_board,
            move=m,
            parent=node,
            is_maximising=not node.is_maximising,
        )
        node.children.append(child)
        return child

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until we hit an unexpanded or terminal node."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _backpropagate(self, node: MCTSNode, score: float):
        """Push simulation result up the tree."""
        while node is not None:
            node.visits += 1
            node.wins += score
            # Flip sign as we go up (alternating player perspective)
            score = -score
            node = node.parent

    def search(self, board, time_budget: float) -> Move:
        """
        Run MCTS for `time_budget` seconds and return the best move.
        """
        root = MCTSNode(board=board.get_copy(), move=None, parent=None, is_maximising=True)
        root.untried_moves = self._get_moves(root.board)
        random.shuffle(root.untried_moves)

        deadline = time.perf_counter() + time_budget

        while time.perf_counter() < deadline:
            # 1. Selection
            leaf = self._select(root)

            # 2. Expansion
            if not leaf.is_terminal():
                leaf = self._expand(leaf)

            # 3. Simulation
            score = self._rollout(leaf.board)

            # 4. Backpropagation
            self._backpropagate(leaf, score)

        if not root.children:
            # Fallback: no tree built (extremely short budget)
            moves = self._get_moves(board)
            return random.choice(moves) if moves else Move.plain(Direction.UP)

        best = root.best_move_child()
        return best.move


# ─────────────────────────────────────────────
#  Player Agent
# ─────────────────────────────────────────────

class PlayerAgent:
    """
    MCTS-based agent with HMM rat tracking.

    Strategy:
      - Each turn, update the rat belief using HMM (predict + update).
      - If expected value of searching top rat cell is high enough, search.
      - Otherwise, run MCTS to pick the best carpet/prime/plain move.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            # Fallback: uniform random walk
            T = np.ones((64, 64)) / 64.0

        self.rat_tracker = RatTracker(T)
        self.mcts = MCTS(self.rat_tracker, rollout_depth=10)

        self.turn_number = 0
        self._last_commentary = []

    def commentate(self) -> str:
        top_cell, top_prob = self.rat_tracker.top_cell()
        return (
            f"Turns played: {self.turn_number} | "
            f"Top rat belief: {top_cell} at {top_prob:.2%}"
        )

    def play(
        self,
        board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        try:
            return self._play(board, sensor_data, time_left)
        except Exception as e:
            # Safety net: never crash
            moves = board.get_valid_moves()
            return random.choice(moves) if moves else Move.plain(Direction.UP)

    def _play(self, board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turn_number += 1
        noise, dist_obs = sensor_data
        worker_pos = board.player_worker.get_location()
        turns_left = board.player_worker.turns_left

        # ── 1. Update rat belief ──────────────────────────────────────────
        self.rat_tracker.predict()
        self.rat_tracker.update(board, int(noise), int(dist_obs), worker_pos)

        # If opponent searched last turn, update belief accordingly
        opp_loc, opp_hit = board.opponent_search
        if opp_loc is not None:
            if opp_hit:
                self.rat_tracker.mark_caught()
            else:
                self.rat_tracker.mark_missed(opp_loc)

        # ── 2. Decide whether to search for rat ───────────────────────────
        top_cell, top_prob = self.rat_tracker.top_cell()
        search_ev = self.rat_tracker.search_expected_value()

        # Search if EV is positive and belief is concentrated enough.
        # Also be more aggressive about searching as game ends.
        end_game = turns_left <= 5
        should_search = (
            search_ev >= SEARCH_EV_THRESHOLD
            and top_prob >= MIN_BELIEF_TO_SEARCH
        ) or (end_game and search_ev > 0)

        if should_search:
            return Move.search(top_cell)

        # ── 3. MCTS for movement ──────────────────────────────────────────
        # Budget: use at most 15% of remaining time per move,
        # but always leave at least 2 seconds in reserve.
        remaining = time_left()
        max_budget = max(0.05, remaining - 2.0)
        time_budget = min(max_budget, remaining * 0.15)

        return self.mcts.search(board, time_budget)

