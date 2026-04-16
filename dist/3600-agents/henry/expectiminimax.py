import time
from typing import Tuple, List, Dict
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
    is_maximizing: bool, is_player_a: bool, rat_belief, 
    timeout_fn, tt: Dict
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
    moves = order_moves(moves, tt_best, rat_belief)

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
                eval_hit, _ = expectiminimax(board_hit, depth - 1, alpha, beta, False, is_player_a, rat_belief, timeout_fn, tt)
                
                # Branch 2: Search Miss
                board_miss = child_board.get_copy()
                board_miss.player_worker.decrement_points(RAT_PENALTY)
                board_miss.reverse_perspective()
                eval_miss, _ = expectiminimax(board_miss, depth - 1, alpha, beta, False, is_player_a, rat_belief, timeout_fn, tt)
                
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, False, is_player_a, rat_belief, timeout_fn, tt)
                
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
                
                board_hit = child_board.get_copy()
                board_hit.player_worker.increment_points(RAT_BONUS)
                board_hit.reverse_perspective()
                eval_hit, _ = expectiminimax(board_hit, depth - 1, alpha, beta, True, is_player_a, rat_belief, timeout_fn, tt)
                
                board_miss = child_board.get_copy()
                board_miss.player_worker.decrement_points(RAT_PENALTY)
                board_miss.reverse_perspective()
                eval_miss, _ = expectiminimax(board_miss, depth - 1, alpha, beta, True, is_player_a, rat_belief, timeout_fn, tt)
                
                eval_score = (p_hit * eval_hit) + (p_miss * eval_miss)
            else:
                child_board.reverse_perspective()
                eval_score, _ = expectiminimax(child_board, depth - 1, alpha, beta, True, is_player_a, rat_belief, timeout_fn, tt)

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

def get_best_move(board: Board, max_time: float, is_player_a: bool, rat_belief) -> Move:
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
                True, is_player_a, rat_belief, is_timeout, tt
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