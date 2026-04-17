import time
import numpy as np
from typing import Tuple, List, Dict
from game.board import Board
from game.move import Move
from game.enums import BOARD_SIZE, RAT_BONUS, RAT_PENALTY
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

def order_moves(moves: List[Move], tt_best_move: Move, rat_belief) -> List[Move]:
    """
    Orders moves to maximize alpha-beta pruning efficiency. 
    Prioritizes Transposition Table best moves, high-scoring carpets, and probable searches.
    """
    def score_move(move: Move) -> float:
        if (tt_best_move and move.move_type == tt_best_move.move_type 
            and move.direction == tt_best_move.direction 
            and move.roll_length == tt_best_move.roll_length 
            and move.search_loc == tt_best_move.search_loc):
            return 10000.0  # Highest priority for previously discovered best move
            
        if move.move_type == MoveType.CARPET:
            return 100.0 * move.roll_length
        elif move.move_type == MoveType.SEARCH:
            if rat_belief is not None:
                p = rat_belief[move.search_loc[1] * BOARD_SIZE + move.search_loc[0]]
                return 500.0 * p
            return 0.0
        elif move.move_type == MoveType.PRIME:
            return 10.0
        return 0.0
    
    return sorted(moves, key=score_move, reverse=True)

def expectiminimax(
    board: Board, depth: int, alpha: float, beta: float, 
    is_maximizing: bool, rat_belief, 
    timeout_fn, tt: Dict
) -> Tuple[float, Move]:
    """
    Core recursive search function. Search moves are excluded from the tree
    entirely — they are handled as a separate decision at the root in
    get_best_move(). This keeps branching factor tight so iterative deepening
    reaches meaningful depths within the time budget.
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
        return evaluate_board(board, rat_belief), None

    # Exclude search moves — handled at root level only
    moves = board.get_valid_moves(enemy=False, exclude_search=True)
    tt_best = tt_entry['best_move'] if tt_entry else None
    moves = order_moves(moves, tt_best, rat_belief)

    best_move = None
    original_alpha = alpha
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue
            child_board.reverse_perspective()
            eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, False, rat_belief, timeout_fn, tt)
                
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
        original_beta = beta
        
        for move in moves:
            child_board = board.forecast_move(move)
            if not child_board: 
                continue
            child_board.reverse_perspective()
            eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, True, rat_belief, timeout_fn, tt)

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

def get_best_move(board: Board, max_time: float, rat_belief) -> Move:
    """
    Entry point for the search algorithm. Uses Iterative Deepening to safely 
    maximize search depth within the allocated time budget.

    Search moves are evaluated here at the root as a simple EV comparison
    against the best movement score, keeping the tree branching factor tight.
    """
    tt = {}
    best_move = None
    start_time = time.time()
    
    def is_timeout():
        return (time.time() - start_time) >= (max_time - 0.05)

    # Iterative deepening over movement moves only
    for depth in range(1, 100):
        try:
            score, move = expectiminimax(
                board, depth, float('-inf'), float('inf'), 
                True, rat_belief, is_timeout, tt
            )
            if move is not None:
                best_move = move
        except TimeoutException:
            break

    # Fallback: any valid movement move
    if best_move is None:
        valid_moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if valid_moves:
            best_move = valid_moves[0]

    # --- Root-level search decision ---
    # Compare EV of searching the best cell vs. making a movement move.
    # EV(search) = p*4 - (1-p)*2 = 6p - 2; positive when p > 1/3.
    if rat_belief is not None:
        best_idx = int(np.argmax(rat_belief))
        p = float(rat_belief[best_idx])
        search_ev = p * 4.0 - (1.0 - p) * 2.0
        # Convert search_ev to heuristic units to compare with tree score
        if search_ev > 0 and search_ev * 50.0 > (score if best_move is not None else float('-inf')):
            rx, ry = best_idx % BOARD_SIZE, best_idx // BOARD_SIZE
            return Move.search((rx, ry))

    return best_move