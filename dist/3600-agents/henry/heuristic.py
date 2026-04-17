import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

def _compute_reachability(buildable_mask: int, px: int, py: int, ox: int, oy: int) -> float:
    """
    Standard NumPy function for fast vectorized reachability evaluation.
    Executes the 4-step pipeline: Binarization, Line Detection, Entry Extraction, and Reachability.
    """
    # 1. State Binarization via Bitboard Extraction
    shifts = np.arange(64, dtype=np.uint64)
    bits = (np.uint64(buildable_mask) >> shifts) & np.uint64(1)
    buildable = bits.reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)
    
    # 2. Vectorized Line Detection (Length >= 2)
    horiz_lines = buildable[:, :-1] & buildable[:, 1:]
    vert_lines = buildable[:-1, :] & buildable[1:, :]
    
    # 3. Entry Point Extraction
    # Standard NumPy allows direct slice assignment
    left_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    left_entries[:, :-2] = horiz_lines[:, 1:]
    
    right_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    right_entries[:, 2:] = horiz_lines[:, :-1]
    
    top_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    top_entries[:-2, :] = vert_lines[1:, :]
    
    bottom_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    bottom_entries[2:, :] = vert_lines[:-1, :]
    
    # Consolidate entries and ensure they lie on buildable (SPACE or PRIMED) terrain
    valid_entries = (left_entries | right_entries | top_entries | bottom_entries) & buildable
    
    # 4. Graph Reachability Race
    y_coords, x_coords = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    
    # Calculate Manhattan Distances
    dist_p = np.abs(x_coords - px) + np.abs(y_coords - py)
    dist_o = np.abs(x_coords - ox) + np.abs(y_coords - oy)
    
    # Territory control weighted by proximity (adding epsilon to prevent division by zero)
    score_p = np.sum(valid_entries / (dist_p + 1.0))
    score_o = np.sum(valid_entries / (dist_o + 1.0))
    
    return float(score_p - score_o)

def evaluate_board(board: Board, is_player_a: bool, rat_belief: np.ndarray = None) -> float:
    """
    Scores the board state utilizing vectorized reachability metrics.
    Returns a positive float if the state favors the requesting player, negative if it favors the opponent.
    """
    if board.is_player_a_turn == is_player_a:
        my_worker = board.player_worker
        opp_worker = board.opponent_worker
    else:
        my_worker = board.opponent_worker
        opp_worker = board.player_worker
        
    my_loc = my_worker.get_location()
    opp_loc = opp_worker.get_location()
        
    # 1. Point Differential (Primary Objective)
    score = (my_worker.get_points() - opp_worker.get_points()) * 75.0
    
    # 2. Vectorized Board Potential
    buildable_mask = board._space_mask | board._primed_mask
    
    reachability_diff = _compute_reachability(
        buildable_mask, 
        my_loc[0], my_loc[1], 
        opp_loc[0], opp_loc[1]
    )
    
    score += reachability_diff * 25.0
    
    # 3. Rat Hunting Potential
    if rat_belief is not None:
        best_rat_idx = int(np.argmax(rat_belief))
        rx = best_rat_idx % BOARD_SIZE
        ry = best_rat_idx // BOARD_SIZE
        max_prob = rat_belief[best_rat_idx]
        
        if max_prob > 0.15:
            dist_me_rat = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
            dist_opp_rat = abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry)
            
            expected_rat_value = max_prob * 100 
            
            score += expected_rat_value / (dist_me_rat + 1.0)
            score -= expected_rat_value / (dist_opp_rat + 1.0)
            
    return score