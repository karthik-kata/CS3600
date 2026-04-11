import time
import random
from collections.abc import Callable
from typing import Tuple, List

from game import board, move, enums

class PlayerAgent:
    def __init__(self, board: board.Board, transition_matrix=None, time_left: Callable = None):
        # Center control heatmap
        # Promotes controlling the center of the 8x8 grid where mobility is highest.
        self.center_heatmap = [
            0, 0, 1, 1, 1, 1, 0, 0,
            0, 1, 2, 2, 2, 2, 1, 0,
            1, 2, 3, 3, 3, 3, 2, 1,
            1, 2, 3, 4, 4, 3, 2, 1,
            1, 2, 3, 4, 4, 3, 2, 1,
            1, 2, 3, 3, 3, 3, 2, 1,
            0, 1, 2, 2, 2, 2, 1, 0,
            0, 0, 1, 1, 1, 1, 0, 0
        ]
        
        self.points_table = enums.CARPET_POINTS_TABLE
        self.transposition_table = {}

    def play(self, game_board: board.Board, sensor_data: Tuple, time_left: Callable):
        start_time = time.time()
        time_limit = 1.8  # Soft limit to ensure we return a move in ~2 seconds
        
        best_move = None
        current_depth = 1
        
        # Initial Move Ordering to maximize Alpha-Beta pruning efficiency
        legal_moves = game_board.get_valid_moves()
        legal_moves.sort(key=self._heuristic_move_weight, reverse=True)
        
        if not legal_moves:
            return None

        # Iterative Deepening Loop
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
                
            try:
                move_found = self._perform_search(game_board, current_depth, start_time, time_limit)
                if move_found:
                    best_move = move_found
                current_depth += 1
            except TimeoutError:
                break  # Stop searching and use the best move from the last fully completed depth
        
        return best_move if best_move else random.choice(legal_moves)

    def _perform_search(self, state: board.Board, depth: int, start_time: float, limit: float):
        alpha = float('-inf')
        beta = float('inf')
        best_m = None
        
        moves = state.get_valid_moves()
        moves.sort(key=self._heuristic_move_weight, reverse=True)
        
        max_val = float('-inf')
        for m in moves:
            if (time.time() - start_time) > limit:
                raise TimeoutError()
                
            sim_board = state.forecast_move(m)
            if sim_board:
                # Reverse perspective transforms the board so the opponent is now the "player_worker"
                # This allows us to use standard Negamax logic.
                sim_board.reverse_perspective()
                val = -self._minimax(sim_board, depth - 1, -beta, -alpha, start_time, limit)
                
                if val > max_val:
                    max_val = val
                    best_m = m
                alpha = max(alpha, val)
                
        return best_m

    def _minimax(self, state: board.Board, depth: int, alpha: float, beta: float, start_time: float, limit: float):
        if (time.time() - start_time) > limit:
            raise TimeoutError()

        if depth == 0 or state.is_game_over():
            return self.evaluate_board(state)

        # Transposition Table Lookup using a hash of the critical bitboards and worker positions
        state_key = (
            state._primed_mask, 
            state._carpet_mask, 
            state.player_worker.get_location(), 
            state.opponent_worker.get_location()
        )
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        moves = state.get_valid_moves()
        moves.sort(key=self._heuristic_move_weight, reverse=True)

        max_eval = float('-inf')
        for m in moves:
            sim = state.forecast_move(m)
            if sim:
                sim.reverse_perspective()
                eval = -self._minimax(sim, depth - 1, -beta, -alpha, start_time, limit)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        
        self.transposition_table[state_key] = max_eval
        return max_eval

    def _heuristic_move_weight(self, m: move.Move):
        """Static evaluation of moves for move ordering."""
        if m.move_type == enums.MoveType.CARPET:
            return 100 + self.points_table.get(m.roll_length, 0)
        if m.move_type == enums.MoveType.PRIME:
            return 50
        return 0

    def evaluate_board(self, state: board.Board) -> float:
        """
        Advanced Heuristic combining Score, Mobility, Center Control, and Bitwise Line Detection.
        Evaluates from the perspective of state.player_worker (the player whose turn it currently is).
        """
        # 1. Score Delta (Crucial for Zero-Sum games)
        my_pts = state.player_worker.get_points()
        opp_pts = state.opponent_worker.get_points()
        score_delta = (my_pts - opp_pts) * 1000.0

        # 2. Mobility
        my_moves_count = len(state.get_valid_moves(enemy=False))
        opp_moves_count = len(state.get_valid_moves(enemy=True))
        mobility_delta = (my_moves_count - opp_moves_count) * 10.0

        # 3. Center Control Heatmap
        my_pos = state.player_worker.get_location()
        opp_pos = state.opponent_worker.get_location()
        my_center = self.center_heatmap[my_pos[1] * 8 + my_pos[0]]
        opp_center = self.center_heatmap[opp_pos[1] * 8 + opp_pos[0]]
        center_delta = (my_center - opp_center) * 5.0

        # 4. Bitwise Prime Line Detection & Threat Evaluation
        # Instead of generic board primes, evaluate immediate carpet potential for both sides
        my_carpet_threat = self._evaluate_immediate_carpet_threat(state, is_enemy=False)
        opp_carpet_threat = self._evaluate_immediate_carpet_threat(state, is_enemy=True)

        # 5. Global Prime Connectivity Tension
        # Rewards the board state if there are long interconnected lines of primes
        # We apply a slight asymmetric multiplier based on who is closer to the primes using adjacency bitmasks
        global_tension = self.evaluate_prime_lines_advanced(state._primed_mask)
        tension_delta = global_tension * (1.5 if my_carpet_threat > opp_carpet_threat else -1.5)

        # Heavily penalize the state if the opponent has a massive carpet threat next turn
        threat_penalty = -500.0 if opp_carpet_threat >= 10 else 0.0

        return score_delta + mobility_delta + center_delta + (my_carpet_threat * 10) - (opp_carpet_threat * 10) + tension_delta + threat_penalty

    def _evaluate_immediate_carpet_threat(self, state: board.Board, is_enemy: bool) -> float:
        """Checks the highest-value carpet roll currently available to the given worker."""
        moves = state.get_valid_moves(enemy=is_enemy)
        max_threat = 0.0
        for m in moves:
            if m.move_type == enums.MoveType.CARPET:
                pts = self.points_table.get(m.roll_length, 0)
                if pts > max_threat:
                    max_threat = pts
        return max_threat

    def evaluate_prime_lines_advanced(self, primes_mask: int) -> float:
        """
        Calculates the presence of contiguous prime lines on the 64-bit mask.
        Uses bitwise shifts to detect lines of length 2 to 7.
        """
        score = 0.0
        
        # Check Horizontal Connections (Shift Left by 1)
        h_mask = primes_mask
        for length in range(2, 8):
            # 0xFEFEFEFEFEFEFEFE prevents bits wrapping around the right edge to the left edge of the next row
            h_mask = h_mask & ((h_mask << 1) & 0xFEFEFEFEFEFEFEFE)
            if not h_mask: 
                break
            score += bin(h_mask).count('1') * (length * 1.5)

        # Check Vertical Connections (Shift Left by 8 rows)
        v_mask = primes_mask
        for length in range(2, 8):
            v_mask = v_mask & (v_mask << 8)
            if not v_mask: 
                break
            score += bin(v_mask).count('1') * (length * 1.5)

        return score

    def commentate(self):
        return "fweh!"