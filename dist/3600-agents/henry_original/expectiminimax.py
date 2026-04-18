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

def hash_board(board: Board, rat_belief) -> int:
    """
    Generates a fast, unique hash for the current board state AND the belief distribution.
    The belief state is discretized to prevent float-precision collisions.
    """
    belief_tuple = tuple(np.round(rat_belief, 3)) if rat_belief is not None else None
    
    return hash((
        board._space_mask, board._primed_mask, board._carpet_mask, board._blocked_mask,
        board.player_worker.get_location(), board.opponent_worker.get_location(),
        board.player_worker.get_points(), board.opponent_worker.get_points(),
        board.is_player_a_turn,
        belief_tuple
    ))

def order_moves(
    board: Board, 
    moves: List[Move], 
    tt_best_move: Move, 
    is_player_a: bool, 
    is_maximizing: bool,
    rat_belief
) -> List[Move]:
    """Orders moves to maximize alpha-beta pruning efficiency."""
    def score_move(move: Move) -> float:
        if (tt_best_move and move.move_type == tt_best_move.move_type 
            and move.direction == tt_best_move.direction 
            and move.roll_length == tt_best_move.roll_length 
            and move.search_loc == tt_best_move.search_loc):
            return float('inf') if is_maximizing else float('-inf')
            
        child_board = board.forecast_move(move)
        if not child_board:
            return float('-inf') if is_maximizing else float('inf')
        
        return evaluate_board(child_board, is_player_a, rat_belief)

    return sorted(moves, key=score_move, reverse=is_maximizing)

def expectiminimax(
    board: Board, depth: int, alpha: float, beta: float, 
    is_maximizing: bool, is_player_a: bool, rat_belief, 
    respawn_belief, hmm_trans, timeout_fn, tt: Dict
) -> Tuple[float, Move]:
    """
    Core recursive search function with corrected chance node bounds, 
    belief hashing, aggressive search pruning, and correct HMM propagation.
    """
    if timeout_fn():
        raise TimeoutException()
        
    b_hash = hash_board(board, rat_belief)
    tt_entry = tt.get(b_hash)
    
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
    original_beta = beta
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue

            if move.move_type == MoveType.SEARCH:
                p_hit = rat_belief[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]] if rat_belief is not None else 0.0
                
                # OPTIMIZATION 1: Prune guaranteed search misses
                if p_hit <= 0.15: 
                    continue
                    
                p_miss = 1.0 - p_hit
                eval_hit = 0.0
                eval_miss = 0.0
                
                next_respawn_belief = np.dot(respawn_belief, hmm_trans)
                
                # OPTIMIZATION 2: Short-circuit Hit Branch
                if p_hit > 0.0:
                    board_hit = child_board.get_copy()
                    board_hit.player_worker.increment_points(RAT_BONUS)
                    board_hit.reverse_perspective()
                    # FIX: Passed hmm_trans properly
                    eval_hit, _ = expectiminimax(board_hit, depth - 1, float('-inf'), float('inf'), not is_maximizing, is_player_a, next_respawn_belief, respawn_belief, hmm_trans, timeout_fn, tt)
                
                # OPTIMIZATION 2: Short-circuit Miss Branch
                if p_miss > 0.0:
                    board_miss = child_board.get_copy()
                    board_miss.player_worker.decrement_points(RAT_PENALTY)
                    board_miss.reverse_perspective()
                    
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
                    next_belief_miss = np.dot(rat_belief_miss, hmm_trans) if rat_belief_miss is not None else None
                    
                    # FIX: Passed hmm_trans properly
                    eval_miss, _ = expectiminimax(board_miss, depth - 1, float('-inf'), float('inf'), not is_maximizing, is_player_a, next_belief_miss, respawn_belief, hmm_trans, timeout_fn, tt)
                
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                next_rat_belief = np.dot(rat_belief, hmm_trans) if rat_belief is not None else None
                # FIX: Passed hmm_trans properly
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, not is_maximizing, is_player_a, next_rat_belief, respawn_belief, hmm_trans, timeout_fn, tt)
      
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        
        flag = EXACT
        if max_eval <= original_alpha:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
            
        tt[b_hash] = {'depth': depth, 'flag': flag, 'score': max_eval, 'best_move': best_move}
        return max_eval, best_move
        
    else:
        min_eval = float('inf')
        
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue

            if move.move_type == MoveType.SEARCH:
                p_hit = rat_belief[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]] if rat_belief is not None else 0.0
                
                # OPTIMIZATION 1: Prune guaranteed search misses
                if p_hit <= 0.15: 
                    continue

                p_miss = 1.0 - p_hit
                eval_hit = 0.0
                eval_miss = 0.0
                
                next_respawn_belief = np.dot(respawn_belief, hmm_trans)

                # OPTIMIZATION 2: Short-circuit Hit Branch
                if p_hit > 0.0:
                    board_hit = child_board.get_copy()
                    board_hit.player_worker.increment_points(RAT_BONUS)
                    board_hit.reverse_perspective()
                    # FIX: Passed hmm_trans properly
                    eval_hit, _ = expectiminimax(board_hit, depth - 1, float('-inf'), float('inf'), not is_maximizing, is_player_a, next_respawn_belief, respawn_belief, hmm_trans, timeout_fn, tt)
                
                # OPTIMIZATION 2: Short-circuit Miss Branch
                if p_miss > 0.0:
                    board_miss = child_board.get_copy()
                    board_miss.player_worker.decrement_points(RAT_PENALTY)
                    board_miss.reverse_perspective()
                    
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
                
                    next_belief_miss = np.dot(rat_belief_miss, hmm_trans) if rat_belief_miss is not None else None
                    # FIX: Passed hmm_trans properly
                    eval_miss, _ = expectiminimax(board_miss, depth - 1, float('-inf'), float('inf'), not is_maximizing, is_player_a, next_belief_miss, respawn_belief, hmm_trans, timeout_fn, tt)
                
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                next_rat_belief = np.dot(rat_belief, hmm_trans) if rat_belief is not None else None
                # FIX: Passed next_rat_belief instead of old rat_belief, AND passed hmm_trans properly
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, not is_maximizing, is_player_a, next_rat_belief, respawn_belief, hmm_trans, timeout_fn, tt)

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

# FIX: Changed `hhm_trans` to `hmm_trans`
def get_best_move(board: Board, max_time: float, is_player_a: bool, rat_belief, respawn_belief, hmm_trans) -> Move:
    """Entry point for the search algorithm."""
    tt = {}
    best_move = None
    start_time = time.time()
    
    def is_timeout():
        return (time.time() - start_time) >= (max_time - 0.05)

    for depth in range(1, 100):
        try:
            # FIX: Passed hmm_trans in the correct positional slot
            score, move = expectiminimax(
                board, depth, float('-inf'), float('inf'), 
                True, is_player_a, rat_belief, respawn_belief, hmm_trans, is_timeout, tt
            )
            if move is not None:
                best_move = move
        except TimeoutException:
            break

    if best_move is None:
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=False)
        if valid_moves:
            best_move = valid_moves[0]

    return best_move