import time
from typing import Tuple, List, Dict
import numpy as np
from game.board import Board
from game.move import Move
from game.enums import MoveType, BOARD_SIZE, RAT_BONUS, RAT_PENALTY
from .heuristic import evaluate_board

class TimeoutException(Exception):
    """Raised when the iterative deepening search exceeds the allocated time limit."""
    pass

# Transposition Table Flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

def hash_board(board: Board) -> int:
    """Generates a fast, unique hash for the current board state."""
    return hash((
        board._space_mask, board._primed_mask, board._carpet_mask, board._blocked_mask,
        board.player_worker.get_location(), board.opponent_worker.get_location(),
        board.player_worker.get_points(), board.opponent_worker.get_points(),
        board.is_player_a_turn
    ))

def order_moves(
    board: Board, 
    moves: List[Move], 
    tt_best_move: Move, 
    is_player_a: bool, 
    is_maximizing: bool,
    rat_belief
) -> List[Move]:
    """
    Orders moves to maximize alpha-beta pruning efficiency by previewing 
    the resulting board states with the vectorized heuristic.
    """
    def score_move(move: Move) -> float:
        # 1. Transposition Table Priority (Always King)
        if (tt_best_move and move.move_type == tt_best_move.move_type 
            and move.direction == tt_best_move.direction 
            and move.roll_length == tt_best_move.roll_length 
            and move.search_loc == tt_best_move.search_loc):
            # Give it an astronomically high score so it is always evaluated first
            return float('inf') if is_maximizing else float('-inf')
            
        # 2. Forecast and Evaluate
        child_board = board.forecast_move(move)
        if not child_board:
            # Penalize invalid moves heavily
            return float('-inf') if is_maximizing else float('inf')
        
        base_score = evaluate_board(child_board, is_player_a, rat_belief)
            
        # 3. Apply the NumPy reachability heuristic
        # We don't reverse perspective here because evaluate_board handles 
        # worker assignment internally based on `is_player_a`
        return base_score

    # If it is your turn (maximizing), search the highest scoring moves first.
    # If it is the opponent's turn (minimizing), search the lowest scoring moves first.
    return sorted(moves, key=score_move, reverse=is_maximizing)

def expectiminimax(
    board: Board, depth: int, alpha: float, beta: float, 
    is_maximizing: bool, is_player_a: bool, rat_belief, 
    respawn_belief, timeout_fn, tt: Dict
) -> Tuple[float, Move]:
    """
    Core recursive search function integrating minimax, alpha-beta pruning, 
    and expected value calculations for stochastic SEARCH actions.
    """
    if timeout_fn():
        raise TimeoutException()
        
    b_hash = hash_board(board)
    tt_entry = tt.get(b_hash)
    
    # Transposition Table Lookup
    if tt_entry is not None and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == EXACT:
            return tt_entry['score'], tt_entry['best_move']
        elif tt_entry['flag'] == LOWERBOUND:
            alpha = max(alpha, tt_entry['score'])
        elif tt_entry['flag'] == UPPERBOUND:
            beta = min(beta, tt_entry['score'])
        if alpha >= beta:
            return tt_entry['score'], tt_entry['best_move']
            
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, is_player_a, rat_belief), None

    moves = board.get_valid_moves(enemy=False, exclude_search=False)
    tt_best = tt_entry['best_move'] if tt_entry else None
    moves = order_moves(board, moves, tt_best, is_player_a, is_maximizing, rat_belief)
    best_move = None
    original_alpha = alpha
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue

            # Expecti-node: Branch based on stochastic search outcomes
            if move.move_type == MoveType.SEARCH:
                p_hit = rat_belief[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]] if rat_belief is not None else 0.0
                p_miss = 1.0 - p_hit
                
                # Branch 1: Search Hit
                board_hit = child_board.get_copy()
                board_hit.player_worker.increment_points(RAT_BONUS)
                board_hit.reverse_perspective()
                
                # Pass respawn_belief as the current belief for the remaining depth
                eval_hit, _ = expectiminimax(board_hit, depth - 1, alpha, beta, not is_maximizing, is_player_a, respawn_belief, respawn_belief, timeout_fn, tt)
                
                # Branch 2: Search Miss
                board_miss = child_board.get_copy()
                board_miss.player_worker.decrement_points(RAT_PENALTY)
                board_miss.reverse_perspective()
                
                # Zero out the missed cell and renormalize
                rat_belief_miss = None
                if rat_belief is not None:
                    rat_belief_miss = np.copy(rat_belief)
                    idx = move.search_loc[1] * BOARD_SIZE + move.search_loc[0]
                    rat_belief_miss[idx] = 0.0
                    total_prob = np.sum(rat_belief_miss)
                    if total_prob > 0:
                        rat_belief_miss /= total_prob
                    else:
                        rat_belief_miss = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float64) / (BOARD_SIZE * BOARD_SIZE)
                
                eval_miss, _ = expectiminimax(board_miss, depth - 1, alpha, beta, not is_maximizing, is_player_a, rat_belief_miss, respawn_belief, timeout_fn, tt)
                
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, False, is_player_a, rat_belief, respawn_belief, timeout_fn, tt)
      
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        
        # Transposition Table Storage
        flag = EXACT
        if max_eval <= original_alpha:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
            
        tt[b_hash] = {'depth': depth, 'flag': flag, 'score': max_eval, 'best_move': best_move}
        return max_eval, best_move
        
    else:
        min_eval = float('inf')
        original_beta = beta
        
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue

            if move.move_type == MoveType.SEARCH:
                p_hit = rat_belief[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]] if rat_belief is not None else 0.0
                p_miss = 1.0 - p_hit
                
                # Hit Branch
                board_hit = child_board.get_copy()
                board_hit.player_worker.increment_points(RAT_BONUS)
                board_hit.reverse_perspective()
                eval_hit, _ = expectiminimax(board_hit, depth - 1, alpha, beta, True, is_player_a, respawn_belief, respawn_belief, timeout_fn, tt)
                
                # Miss Branch
                board_miss = child_board.get_copy()
                board_miss.player_worker.decrement_points(RAT_PENALTY)
                board_miss.reverse_perspective()
                
                rat_belief_miss = np.copy(rat_belief) if rat_belief is not None else None
                if rat_belief_miss is not None:
                    rat_belief_miss[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]] = 0.0
                    total = np.sum(rat_belief_miss)
                    if total > 0: rat_belief_miss /= total

                eval_miss, _ = expectiminimax(board_miss, depth - 1, alpha, beta, True, is_player_a, rat_belief_miss, respawn_belief, timeout_fn, tt)
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, True, is_player_a, rat_belief, respawn_belief, timeout_fn, tt)

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
                
        flag = EXACT
        if min_eval <= alpha:
            flag = UPPERBOUND
        elif min_eval >= original_beta:
            flag = LOWERBOUND
            
        tt[b_hash] = {'depth': depth, 'flag': flag, 'score': min_eval, 'best_move': best_move}
        return min_eval, best_move

def get_best_move(board: Board, max_time: float, is_player_a: bool, rat_belief, respawn_belief) -> Move:
    """
    Entry point for the search algorithm. Uses Iterative Deepening to safely 
    maximize search depth within the allocated time constraints.
    """
    tt = {}
    best_move = None
    start_time = time.time()
    
    # 0.05 second safety buffer to ensure we don't break the hard time limit 
    # during deep recursive returns
    def is_timeout():
        return (time.time() - start_time) >= (max_time - 0.05)

    # Iterative deepening
    for depth in range(1, 100):
        try:
            score, move = expectiminimax(
                board, depth, float('-inf'), float('inf'), 
                True, is_player_a, rat_belief, respawn_belief, is_timeout, tt
            )
            if move is not None:
                best_move = move
        except TimeoutException:
            break

    # Fallback default move in case of immediate depth-1 timeout
    if best_move is None:
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=False)
        if valid_moves:
            best_move = valid_moves[0]

    return best_move