from collections.abc import Callable
from typing import List, Tuple
import numpy as np
import random

from game import board as board_module, move as move_module, enums
from game.board import Board
from game.move import Move
from game.enums import (
    Cell, MoveType, Direction, Noise,
    BOARD_SIZE, CARPET_POINTS_TABLE, RAT_BONUS, RAT_PENALTY,
    loc_after_direction
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}
DISTANCE_ERROR_PROBS    = (0.12, 0.70, 0.12, 0.06)
DISTANCE_ERROR_OFFSETS  = (-1, 0, 1, 2)

# Minimax config
MAX_DEPTH       = 3       # expectiminimax ply depth
TIME_SAFETY     = 4.0     # seconds reserved; never think if below this
SEARCH_EV_THRESH = 0.0    # search if EV > this

# Heuristic weights
W_SCORE_DIFF    = 1.0
W_CARPET_POT    = 0.5     # per reachable primed cell that can extend a run
W_RAT_EV        = 1.0     # weight on best search EV term in heuristic


# ─────────────────────────────────────────────
# HMM Rat Belief Tracker
# ─────────────────────────────────────────────
class RatBelief:
    """
    Maintains a probability distribution (belief) over all 64 cells
    representing where the rat currently is.

    Filtering equation each turn:
        predicted = T.T @ belief          (rat moves)
        updated   = predicted * likelihood(obs)
        belief    = updated / sum(updated)
    """

    def __init__(self, T: np.ndarray):
        # T[i, j] = P(rat moves from cell i → cell j)
        self.T = np.array(T, dtype=np.float64)          # (64, 64)
        self.TT = self.T.T                               # transpose for forward pass
        self.belief = np.ones(64, dtype=np.float64) / 64  # uniform prior

        # Build noise likelihood table: shape (64,) per noise type per cell type
        # noise_lut[cell_type_int][noise_int] = probability
        self.noise_lut = {}
        for ct, probs in NOISE_PROBS.items():
            self.noise_lut[int(ct)] = np.array(probs, dtype=np.float64)

    def _cell_types_array(self, board: Board) -> np.ndarray:
        """Returns int array of length 64 with Cell enum value at each cell."""
        ctypes = np.zeros(64, dtype=np.int32)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                ctypes[y * BOARD_SIZE + x] = int(board.get_cell((x, y)))
        return ctypes

    def update(self, board: Board, noise: Noise, noisy_dist: int, worker_pos: Tuple[int, int]):
        """
        Predict + update belief given a new observation.

        Parameters:
            board      : current board (used for cell types)
            noise      : Noise enum (SQUEAK/SCRATCH/SQUEAL)
            noisy_dist : noisy manhattan distance reported
            worker_pos : (x, y) of the observing worker
        """
        # ── Predict: rat moves one step ──
        predicted = self.TT @ self.belief

        # ── Likelihood from noise type ──
        ctypes = self._cell_types_array(board)
        noise_idx = int(noise)
        noise_likelihood = np.array(
            [self.noise_lut[ctypes[i]][noise_idx] for i in range(64)],
            dtype=np.float64
        )

        # ── Likelihood from noisy distance ──
        wx, wy = worker_pos
        dist_likelihood = np.zeros(64, dtype=np.float64)
        for i in range(64):
            ry, rx = divmod(i, BOARD_SIZE)
            actual_dist = abs(rx - wx) + abs(ry - wy)
            p = 0.0
            for offset, prob in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS):
                if noisy_dist == max(0, actual_dist + offset):
                    p += prob
            dist_likelihood[i] = p

        # ── Combine and normalize ──
        updated = predicted * noise_likelihood * dist_likelihood
        total = updated.sum()
        if total > 1e-15:
            self.belief = updated / total
        else:
            # Degenerate: fall back to predict only
            self.belief = predicted / predicted.sum()

    def best_search(self) -> Tuple[Tuple[int, int], float, float]:
        """
        Returns (best_cell, p_rat_there, search_ev).
        search_ev = p * RAT_BONUS - (1-p) * RAT_PENALTY
        """
        best_idx  = int(np.argmax(self.belief))
        p         = float(self.belief[best_idx])
        ev        = p * RAT_BONUS - (1.0 - p) * RAT_PENALTY
        bx        = best_idx % BOARD_SIZE
        by        = best_idx // BOARD_SIZE
        return (bx, by), p, ev

    def rat_caught(self, board: Board):
        """Reset belief to uniform after rat is caught (new rat spawns)."""
        self.belief = np.ones(64, dtype=np.float64) / 64

    def get_belief_grid(self) -> np.ndarray:
        """Return (8, 8) belief array for debugging."""
        return self.belief.reshape(BOARD_SIZE, BOARD_SIZE)


# ─────────────────────────────────────────────
# Heuristic Evaluation
# ─────────────────────────────────────────────
def count_primed_neighbors(board: Board, loc: Tuple[int, int]) -> int:
    """Count primed cells adjacent to loc (potential carpet extension)."""
    count = 0
    for d in Direction:
        nxt = loc_after_direction(loc, d)
        if board.is_valid_cell(nxt) and board.get_cell(nxt) == Cell.PRIMED:
            count += 1
    return count


def carpet_potential(board: Board) -> float:
    """
    Heuristic: for each direction from player's position,
    count consecutive primed cells → look up carpet points.
    """
    loc = board.player_worker.get_location()
    total = 0.0
    for d in Direction:
        cur = loc
        length = 0
        for _ in range(BOARD_SIZE - 1):
            nxt = loc_after_direction(cur, d)
            if not board.is_valid_cell(nxt):
                break
            if board.get_cell(nxt) != Cell.PRIMED:
                break
            length += 1
            cur = nxt
        if length >= 1:
            total += CARPET_POINTS_TABLE.get(length, 0)
    return total


def evaluate(board: Board, belief: RatBelief) -> float:
    """
    Leaf-node heuristic (player's perspective).
    Higher = better for player.
    """
    my_pts  = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    score_diff = my_pts - opp_pts

    cp = carpet_potential(board)

    _, _, rat_ev = belief.best_search()
    rat_term = max(rat_ev, 0.0)  # only count positive EV

    return (W_SCORE_DIFF * score_diff
            + W_CARPET_POT * cp
            + W_RAT_EV * rat_term)


# ─────────────────────────────────────────────
# Expectiminimax
# ─────────────────────────────────────────────
def expectiminimax(board: Board, belief: RatBelief,
                   depth: int, is_max: bool,
                   alpha: float, beta: float) -> float:
    """
    Expectiminimax with alpha-beta pruning.

    - MAX node  : our turn  → maximize
    - MIN node  : opponent's turn → minimize
    - Rat is treated as a background stochastic process embedded in
      the belief; we don't add explicit chance nodes for it here
      (the belief is updated outside the tree).  Inside the tree we
      use the current belief to score search moves.

    Returns heuristic value from the MAX player's perspective.
    """
    if depth == 0 or board.is_game_over():
        return evaluate(board, belief)

    moves = board.get_valid_moves(exclude_search=True)

    # Add search move if its EV is positive
    _, _, sev = belief.best_search()
    if sev > SEARCH_EV_THRESH:
        best_cell, _, _ = belief.best_search()
        moves.append(Move.search(best_cell))

    if not moves:
        return evaluate(board, belief)

    if is_max:
        value = -1e9
        for mv in moves:
            if mv.move_type == MoveType.SEARCH:
                # Chance node for search: EV = p*win_delta + (1-p)*loss_delta
                best_cell, p, _ = belief.best_search()
                child = board.get_copy()
                child.player_worker.increment_points(
                    int(round(p * RAT_BONUS - (1 - p) * RAT_PENALTY))
                )
                child.end_turn()
                child.reverse_perspective()
                val = p * (evaluate(board, belief) + RAT_BONUS) + \
                      (1 - p) * (evaluate(board, belief) - RAT_PENALTY)
            else:
                child = board.forecast_move(mv, check_ok=False)
                if child is None:
                    continue
                child.reverse_perspective()
                val = expectiminimax(child, belief, depth - 1, False, alpha, beta)

            if val > value:
                value = val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
    else:
        value = 1e9
        for mv in moves:
            child = board.forecast_move(mv, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = expectiminimax(child, belief, depth - 1, True, alpha, beta)

            if val < value:
                value = val
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value


def pick_move(board: Board, belief: RatBelief, time_left: Callable, depth: int = MAX_DEPTH) -> Move:
    """
    Root-level move selection via expectiminimax.
    Falls back to shallower depth or greedy if time is low.
    """
    if time_left() < TIME_SAFETY:
        return _greedy_move(board, belief)

    moves = board.get_valid_moves(exclude_search=True)

    # Consider search if EV is positive
    best_cell, p_rat, sev = belief.best_search()
    include_search = sev > SEARCH_EV_THRESH
    if include_search:
        moves.append(Move.search(best_cell))

    if not moves:
        return random.choice(board.get_valid_moves(exclude_search=False))

    best_move  = moves[0]
    best_value = -1e9

    for mv in moves:
        if time_left() < TIME_SAFETY:
            break

        if mv.move_type == MoveType.SEARCH:
            # Expected point gain from search, projected onto the eval scale
            expected_pts = p_rat * RAT_BONUS - (1 - p_rat) * RAT_PENALTY
            val = W_SCORE_DIFF * expected_pts + W_RAT_EV * max(sev, 0.0)
        else:
            child = board.forecast_move(mv, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = expectiminimax(child, belief, depth - 1, False, -1e9, 1e9)

        if val > best_value:
            best_value = val
            best_move  = mv

    return best_move


def _greedy_move(board: Board, belief: RatBelief) -> Move:
    """Fast greedy fallback: pick the immediately highest-value move."""
    moves = board.get_valid_moves(exclude_search=True)
    best_cell, p_rat, sev = belief.best_search()

    best_move  = None
    best_score = -1e9

    for mv in moves:
        child = board.forecast_move(mv, check_ok=False)
        if child is None:
            continue
        sc = child.player_worker.get_points() - child.opponent_worker.get_points()
        if sc > best_score:
            best_score = sc
            best_move  = mv

    if sev > best_score and sev > SEARCH_EV_THRESH:
        return Move.search(best_cell)

    if best_move is None:
        all_moves = board.get_valid_moves(exclude_search=False)
        return random.choice(all_moves) if all_moves else Move.search((0, 0))

    return best_move


# ─────────────────────────────────────────────
# Player Agent
# ─────────────────────────────────────────────
class PlayerAgent:
    """
    Expectiminimax agent with HMM rat belief tracking.

    Strategy:
      - Maintain a Bayesian belief over rat location (HMM filter).
      - Each turn, run expectiminimax to depth MAX_DEPTH.
      - Score leaf nodes with: score_diff + carpet_potential + rat_search_EV.
      - Search for rat when expected value is positive.
    """

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            # Fallback uniform random walk (shouldn't happen in real games)
            T = np.ones((64, 64), dtype=np.float64) / 64

        self.belief = RatBelief(T)
        self.turn   = 0
        self.stats  = {"searches": 0, "hits": 0, "misses": 0}

    def commentate(self):
        s = self.stats
        hit_rate = s["hits"] / s["searches"] if s["searches"] else 0.0
        return (
            f"Searches: {s['searches']}  Hits: {s['hits']}  "
            f"Misses: {s['misses']}  Hit-rate: {hit_rate:.2%}"
        )

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        """
        Called each turn. Returns a Move.

        sensor_data = (noise: Noise, estimated_distance: int)
        """
        noise, noisy_dist = sensor_data
        self.turn += 1

        # ── Update rat belief with this turn's sensor ──
        worker_pos = board.player_worker.get_location()
        self.belief.update(board, noise, noisy_dist, worker_pos)

        # ── Also incorporate opponent's last search result ──
        opp_search_loc, opp_search_result = board.opponent_search
        if opp_search_loc is not None:
            idx = opp_search_loc[1] * BOARD_SIZE + opp_search_loc[0]
            if opp_search_result:
                # Rat was caught there → reset belief (new rat spawned)
                self.belief.rat_caught(board)
            else:
                # Rat was NOT there → zero out that cell
                self.belief.belief[idx] = 0.0
                s = self.belief.belief.sum()
                if s > 1e-15:
                    self.belief.belief /= s

        # ── Also incorporate our last search result ──
        my_search_loc, my_search_result = board.player_search
        if my_search_loc is not None:
            idx = my_search_loc[1] * BOARD_SIZE + my_search_loc[0]
            if my_search_result:
                self.belief.rat_caught(board)
                self.stats["hits"] += 1
            else:
                self.belief.belief[idx] = 0.0
                s = self.belief.belief.sum()
                if s > 1e-15:
                    self.belief.belief /= s
                self.stats["misses"] += 1

        # ── Adaptive depth based on time remaining ──
        t = time_left()
        if t > 30:
            depth = MAX_DEPTH
        elif t > 10:
            depth = 2
        else:
            depth = 1

        # ── Pick and return move ──
        chosen = pick_move(board, self.belief, time_left, depth)

        if chosen.move_type == MoveType.SEARCH:
            self.stats["searches"] += 1

        return chosen
