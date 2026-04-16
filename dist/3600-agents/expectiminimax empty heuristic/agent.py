from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np

from game import board as board_module, move as move_module, enums
from game.enums import (
    MoveType, Direction, Cell, Noise, BOARD_SIZE,
    DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE
)
from game.move import Move
from game.board import Board


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

# Expectiminimax search depth. 2 is fast; 3 is stronger but slower.
SEARCH_DEPTH = 2


# ---------------------------------------------------------------------------
# HMM Rat Belief Tracker
# ---------------------------------------------------------------------------

class RatBelief:
    """
    Maintains a probability distribution over all 64 cells for the rat's
    location using a Bayesian HMM filter.

    State vector: belief[i] = P(rat at cell i), length 64, sums to 1.
    Index mapping: i = y * BOARD_SIZE + x  =>  (x, y) = (i % 8, i // 8)
    """

    def __init__(self, T: np.ndarray):
        """
        Parameters
        ----------
        T : np.ndarray, shape (64, 64)
            T[i, j] = P(rat moves from cell i to cell j).
        """
        self.T = np.array(T, dtype=np.float64)          # (64, 64)
        self.T_t = self.T.T.copy()                       # (64, 64) transposed for fast predict
        self.belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
        self.belief /= self.belief.sum()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, board: Board, noise: Noise, estimated_distance: int):
        """
        Predict + update the belief given this turn's sensor readings.

        Call this at the start of every play() turn BEFORE doing anything
        else, so that the belief reflects the rat's state right now.

        Parameters
        ----------
        board             : current Board (used to read cell types)
        noise             : Noise enum value heard this turn
        estimated_distance: noisy Manhattan distance reported by sensor
        """
        # 1. PREDICT: propagate belief through transition model
        #    belief_pred[j] = sum_i belief[i] * T[i,j] = T^T @ belief
        belief_pred = self.T_t @ self.belief

        # 2. UPDATE: reweight by likelihood of (noise, distance) observation
        likelihood = self._observation_likelihood(board, noise, estimated_distance)

        belief_updated = belief_pred * likelihood

        # Normalise (guard against numerical zero)
        total = belief_updated.sum()
        if total > 1e-12:
            self.belief = belief_updated / total
        else:
            # Complete degeneracy: reset to uniform
            self.belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
            self.belief /= self.belief.sum()

    def update_after_catch(self):
        """
        Call this when the rat was caught (by either player).
        Resets to uniform because a fresh rat spawns at (0,0) and walks
        1000 steps — we have no information about where it ends up.
        """
        self.belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)
        self.belief /= self.belief.sum()

    def update_after_failed_search(self, loc: Tuple[int, int]):
        """
        Call this after a failed search at `loc` (either player).
        Zeroes out that cell and renormalises.
        """
        idx = self._loc_to_idx(loc)
        self.belief[idx] = 0.0
        total = self.belief.sum()
        if total > 1e-12:
            self.belief /= total

    def search_ev(self, loc: Tuple[int, int]) -> float:
        """
        Expected value of searching at `loc`.

        EV = P(rat there) * RAT_BONUS + P(rat not there) * (-RAT_PENALTY)
           = p * 4 - (1-p) * 2
           = 6p - 2
        """
        p = self.belief[self._loc_to_idx(loc)]
        return RAT_BONUS * p - RAT_PENALTY * (1.0 - p)

    def best_search(self) -> Tuple[Tuple[int, int], float]:
        """Returns (location, expected_value) for the highest-EV search."""
        best_idx = int(np.argmax(self.belief))
        best_loc = self._idx_to_loc(best_idx)
        return best_loc, self.search_ev(best_loc)

    def prob_at(self, loc: Tuple[int, int]) -> float:
        return float(self.belief[self._loc_to_idx(loc)])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _observation_likelihood(
        self, board: Board, noise: Noise, estimated_distance: int
    ) -> np.ndarray:
        """
        Returns a length-64 array: likelihood[i] = P(obs | rat at cell i).

        obs = (noise_type, estimated_distance) from the *player's* worker.
        """
        player_loc = board.player_worker.get_location()
        likelihood = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float64)

        noise_idx = int(noise)  # 0=squeak, 1=scratch, 2=squeal

        for i in range(BOARD_SIZE * BOARD_SIZE):
            rat_loc = self._idx_to_loc(i)

            # ---- Noise likelihood ----
            cell_type = board.get_cell(rat_loc)
            noise_probs = NOISE_PROBS.get(cell_type, NOISE_PROBS[Cell.SPACE])
            p_noise = noise_probs[noise_idx]

            # ---- Distance likelihood ----
            actual_dist = abs(player_loc[0] - rat_loc[0]) + abs(player_loc[1] - rat_loc[1])
            p_dist = 0.0
            for offset, prob in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS):
                reported = actual_dist + offset
                reported = max(reported, 0)   # clamp as the game does
                if reported == estimated_distance:
                    p_dist += prob

            likelihood[i] = p_noise * p_dist

        return likelihood

    @staticmethod
    def _loc_to_idx(loc: Tuple[int, int]) -> int:
        return loc[1] * BOARD_SIZE + loc[0]

    @staticmethod
    def _idx_to_loc(idx: int) -> Tuple[int, int]:
        return (idx % BOARD_SIZE, idx // BOARD_SIZE)


# ---------------------------------------------------------------------------
# Helpers used inside the tree
# ---------------------------------------------------------------------------

def _loc_to_idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _best_search(belief: np.ndarray) -> Tuple[Tuple[int, int], float]:
    best_idx = int(np.argmax(belief))
    best_loc = (best_idx % BOARD_SIZE, best_idx // BOARD_SIZE)
    best_p = float(belief[best_idx])
    ev = RAT_BONUS * best_p - RAT_PENALTY * (1.0 - best_p)
    return best_loc, ev


def _predict_belief(belief: np.ndarray, T_t: np.ndarray) -> np.ndarray:
    """One-step rat transition (predict only, no sensor update)."""
    b = T_t @ belief
    total = b.sum()
    return b / total if total > 1e-12 else belief.copy()


# ---------------------------------------------------------------------------
# Heuristic  — FILL THIS IN
# ---------------------------------------------------------------------------

def heuristic(board: Board, belief: np.ndarray) -> float:
    """
    Evaluate a board position from the perspective of board.player_worker.

    Parameters
    ----------
    board  : Board from the maximising player's perspective.
             board.player_worker  = you
             board.opponent_worker = opponent
    belief : length-64 numpy array; belief[i] = P(rat at cell i).
             Reflects the rat's predicted state at this node of the tree.

    Return
    ------
    float — higher is better for the player (maximiser).

    Ideas to consider
    -----------------
    - Score delta:
          board.player_worker.get_points() - board.opponent_worker.get_points()

    - Carpet potential: count primed cells within N steps of your worker
      (more primed cells reachable → higher carpet payoff available).

    - Expected value of the best search:
          best_loc, best_ev = _best_search(belief)   # already computed for you

    - Turns remaining as a scaling factor (aggressive early, greedy late).

    - Opponent proximity to your primed chains (risk of them carpeting instead).
    """
    raise NotImplementedError("Fill in heuristic() before running.")


# ---------------------------------------------------------------------------
# Expectiminimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def expectiminimax(
    board: Board,
    belief: np.ndarray,
    T_t: np.ndarray,
    depth: int,
    is_maximising: bool,
    alpha: float,
    beta: float,
) -> float:
    """
    Expectiminimax with alpha-beta pruning.

    The "chance" element is the rat: between plies we advance the belief
    by one predict step (no real observation is available inside the tree)
    so deeper nodes account for rat drift.

    Convention
    ----------
    - Maximising player  = the agent who called play() (always
      board.player_worker from their perspective).
    - After forecast_move() + reverse_perspective() the roles swap,
      so we flip is_maximising at each ply.
    - board is NOT modified — we use forecast_move() which copies.

    Returns
    -------
    float : estimated value of the position for the maximising player.
    """
    if depth == 0 or board.is_game_over():
        return heuristic(board, belief)

    moves = board.get_valid_moves(exclude_search=True)

    # Check whether a search move should be considered at this node
    best_search_loc, best_search_ev = _best_search(belief)
    include_search = best_search_ev > 0

    if is_maximising:
        value = float('-inf')

        # --- Regular moves ---
        for move in moves:
            child = board.forecast_move(move, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            child_belief = _predict_belief(belief, T_t)
            score = expectiminimax(child, child_belief, T_t, depth - 1, False, alpha, beta)
            if score > value:
                value = score
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # beta cut-off

        # --- Search as a chance node ---
        # Expected value = P(hit)*V(hit branch) + P(miss)*V(miss branch)
        if include_search:
            p_hit = float(belief[_loc_to_idx(best_search_loc)])

            # Hit branch: player gains RAT_BONUS, belief resets to uniform
            hit_board = board.get_copy()
            hit_board.player_worker.increment_points(RAT_BONUS)
            hit_board.end_turn()
            hit_board.reverse_perspective()
            hit_belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64) / (BOARD_SIZE * BOARD_SIZE)
            hit_belief = _predict_belief(hit_belief, T_t)
            hit_val = expectiminimax(hit_board, hit_belief, T_t, depth - 1, False, alpha, beta)

            # Miss branch: player loses RAT_PENALTY, zero out searched cell
            miss_board = board.get_copy()
            miss_board.player_worker.decrement_points(RAT_PENALTY)
            miss_board.end_turn()
            miss_board.reverse_perspective()
            miss_belief = belief.copy()
            miss_belief[_loc_to_idx(best_search_loc)] = 0.0
            s = miss_belief.sum()
            miss_belief = miss_belief / s if s > 1e-12 else miss_belief
            miss_belief = _predict_belief(miss_belief, T_t)
            miss_val = expectiminimax(miss_board, miss_belief, T_t, depth - 1, False, alpha, beta)

            search_val = p_hit * hit_val + (1.0 - p_hit) * miss_val
            value = max(value, search_val)

        return value if value != float('-inf') else heuristic(board, belief)

    else:  # minimising (opponent's ply)
        value = float('inf')

        # --- Regular moves ---
        for move in moves:
            child = board.forecast_move(move, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            child_belief = _predict_belief(belief, T_t)
            score = expectiminimax(child, child_belief, T_t, depth - 1, True, alpha, beta)
            if score < value:
                value = score
            beta = min(beta, value)
            if beta <= alpha:
                break  # alpha cut-off

        # --- Opponent search as chance node ---
        # From the maximiser's perspective, the opponent gaining points is bad.
        if include_search:
            p_hit = float(belief[_loc_to_idx(best_search_loc)])

            hit_board = board.get_copy()
            hit_board.player_worker.increment_points(RAT_BONUS)  # opponent (player_worker here) gains
            hit_board.end_turn()
            hit_board.reverse_perspective()
            hit_belief = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64) / (BOARD_SIZE * BOARD_SIZE)
            hit_belief = _predict_belief(hit_belief, T_t)
            hit_val = expectiminimax(hit_board, hit_belief, T_t, depth - 1, True, alpha, beta)

            miss_board = board.get_copy()
            miss_board.player_worker.decrement_points(RAT_PENALTY)
            miss_board.end_turn()
            miss_board.reverse_perspective()
            miss_belief = belief.copy()
            miss_belief[_loc_to_idx(best_search_loc)] = 0.0
            s = miss_belief.sum()
            miss_belief = miss_belief / s if s > 1e-12 else miss_belief
            miss_belief = _predict_belief(miss_belief, T_t)
            miss_val = expectiminimax(miss_board, miss_belief, T_t, depth - 1, True, alpha, beta)

            search_val = p_hit * hit_val + (1.0 - p_hit) * miss_val
            value = min(value, search_val)

        return value if value != float('inf') else heuristic(board, belief)


# ---------------------------------------------------------------------------
# PlayerAgent
# ---------------------------------------------------------------------------

class PlayerAgent:
    """
    Entry points (do not rename): __init__, commentate, play.
    """

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            # Fallback: uniform (shouldn't happen in real games)
            n = BOARD_SIZE * BOARD_SIZE
            T = np.ones((n, n), dtype=np.float64) / n

        self.rat_belief = RatBelief(T)
        self.T_t = self.rat_belief.T_t

        # Stats for commentate()
        self.turns_played = 0
        self.searches_made = 0

    # ------------------------------------------------------------------

    def commentate(self) -> str:
        return f"Turns played: {self.turns_played} | Searches made: {self.searches_made}"

    # ------------------------------------------------------------------

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        """
        Called each turn.
        sensor_data = (noise: Noise, estimated_distance: int)
        """
        noise, estimated_distance = sensor_data
        self.turns_played += 1

        # ---- Incorporate opponent's last search result -----------------
        self._handle_opponent_search(board)

        # ---- Update HMM belief with this turn's sensor readings --------
        self.rat_belief.update(board, noise, estimated_distance)

        # ---- Candidate moves -------------------------------------------
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            # No movement possible — search unconditionally
            best_search_loc, _ = self.rat_belief.best_search()
            self.searches_made += 1
            return Move.search(best_search_loc)

        # ---- Run expectiminimax for each root move ---------------------
        best_move: Optional[Move] = None
        best_score = float('-inf')

        for move in moves:
            child = board.forecast_move(move, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            child_belief = _predict_belief(self.rat_belief.belief, self.T_t)
            score = expectiminimax(
                child,
                child_belief,
                self.T_t,
                depth=SEARCH_DEPTH - 1,   # we already took one ply
                is_maximising=False,       # opponent's turn next
                alpha=float('-inf'),
                beta=float('inf'),
            )
            if score > best_score:
                best_score = score
                best_move = move

        # ---- Compare best movement score vs. searching -----------------
        best_search_loc, best_search_ev = self.rat_belief.best_search()
        if best_search_ev > best_score and best_search_ev > 0:
            self.searches_made += 1
            return Move.search(best_search_loc)

        return best_move if best_move is not None else moves[0]

    # ------------------------------------------------------------------

    def _handle_opponent_search(self, board: Board):
        """Incorporate the opponent's last search result into the belief."""
        opp_loc, opp_result = board.opponent_search
        if opp_loc is None:
            return
        if opp_result:
            self.rat_belief.update_after_catch()
        else:
            self.rat_belief.update_after_failed_search(opp_loc)

