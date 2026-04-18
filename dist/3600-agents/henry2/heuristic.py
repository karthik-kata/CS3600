import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

def _compute_reachability(space_mask: int, primed_mask: int, px: int, py: int, ox: int, oy: int) -> float:
    """
    Standard NumPy function for fast vectorized reachability evaluation.
    Calculates the exact theoretical point yield (priming + carpeting) for all valid lines.
    """
    shifts = np.arange(64, dtype=np.uint64)
    
    # 1. State Binarization via Bitboard Extraction
    space = ((np.uint64(space_mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)
    primed = ((np.uint64(primed_mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)
    buildable = space | primed
    
    # 2. Base Values for Exact Counting
    # L_val will track total line length. S_val will track unprimed spaces.
    L_val = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    S_val = space.astype(np.int32) 
    
    L_h_fwd, L_h_bwd = L_val.copy(), L_val.copy()
    L_v_fwd, L_v_bwd = L_val.copy(), L_val.copy()
    
    S_h_fwd, S_h_bwd = S_val.copy(), S_val.copy()
    S_v_fwd, S_v_bwd = S_val.copy(), S_val.copy()
    
    # 3. Vectorized Line Integration (Counting Lengths)
    for _ in range(7):
        L_h_fwd[:, 1:] = (L_h_fwd[:, :-1] + L_val[:, 1:]) * buildable[:, 1:]
        L_h_bwd[:, :-1] = (L_h_bwd[:, 1:] + L_val[:, :-1]) * buildable[:, :-1]
        L_v_fwd[1:, :] = (L_v_fwd[:-1, :] + L_val[1:, :]) * buildable[1:, :]
        L_v_bwd[:-1, :] = (L_v_bwd[1:, :] + L_val[:-1, :]) * buildable[:-1, :]
        
        S_h_fwd[:, 1:] = (S_h_fwd[:, :-1] + S_val[:, 1:]) * buildable[:, 1:]
        S_h_bwd[:, :-1] = (S_h_bwd[:, 1:] + S_val[:, :-1]) * buildable[:, :-1]
        S_v_fwd[1:, :] = (S_v_fwd[:-1, :] + S_val[1:, :]) * buildable[1:, :]
        S_v_bwd[:-1, :] = (S_v_bwd[1:, :] + S_val[:-1, :]) * buildable[:-1, :]
        
    # 4. Total Contiguous Segment Lengths
    L_h_run = (L_h_fwd + L_h_bwd - L_val) * buildable
    L_v_run = (L_v_fwd + L_v_bwd - L_val) * buildable
    
    S_h_run = (S_h_fwd + S_h_bwd - S_val) * buildable
    S_v_run = (S_v_fwd + S_v_bwd - S_val) * buildable
    
    # A carpet roll maxes out at a length of 7[cite: 40]. 
    # We clip the length here so we don't index out of bounds.
    L_h_run = np.clip(L_h_run, 0, 7)
    L_v_run = np.clip(L_v_run, 0, 7)
    
    # 5. Exact Point Calculation
    # Official Carpet Points Table: Index = Length, Value = Points 
    carpet_points_map = np.array([0, -1, 2, 4, 6, 10, 15, 21], dtype=np.float64)
    
    h_carpet_pts = carpet_points_map[L_h_run]
    v_carpet_pts = carpet_points_map[L_v_run]
    
    # True value = Carpet Points + (+1 point for every unprimed space )
    h_run_value = h_carpet_pts + S_h_run
    v_run_value = v_carpet_pts + S_v_run
    
    # A tile's true potential is its best orientation
    tile_potential = np.maximum(h_run_value, v_run_value)
    
    # 6. Entry Point Extraction
    horiz_lines = buildable[:, :-1] & buildable[:, 1:]
    vert_lines = buildable[:-1, :] & buildable[1:, :]
    
    left_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    left_entries[:, :-2] = horiz_lines[:, 1:]
    
    right_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    right_entries[:, 2:] = horiz_lines[:, :-1]
    
    top_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    top_entries[:-2, :] = vert_lines[1:, :]
    
    bottom_entries = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    bottom_entries[2:, :] = vert_lines[:-1, :]
    
    valid_entries = (left_entries | right_entries | top_entries | bottom_entries) & buildable
    
    entry_values = tile_potential * valid_entries
    
    # 7. Graph Reachability Race
    y_coords, x_coords = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    
    dist_p = np.abs(x_coords - px) + np.abs(y_coords - py)
    dist_o = np.abs(x_coords - ox) + np.abs(y_coords - oy)
    
    score_p = np.sum(entry_values / (dist_p + 1.0))
    score_o = np.sum(entry_values / (dist_o + 1.0))
    
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
    point_diff = (my_worker.get_points() - opp_worker.get_points())
    
    
    # 2. Vectorized Board Potential
    reachability_diff = _compute_reachability(
        board._space_mask, 
        board._primed_mask, 
        my_loc[0], my_loc[1], 
        opp_loc[0], opp_loc[1]
    )
        
    """
    # 3. Rat Hunting Potential
    rat_score = 0
    if rat_belief is not None:
        best_rat_idx = int(np.argmax(rat_belief))
        rx = best_rat_idx % BOARD_SIZE
        ry = best_rat_idx // BOARD_SIZE
        max_prob = rat_belief[best_rat_idx]
        
        # SQUARING max_prob squashes noise. 
        # A 5% chance becomes 0.0025. A 50% chance becomes 0.25.
        # We increase the multiplier (e.g., 1000) to compensate for the smaller fraction.
        expected_rat_value = (max_prob ** 2) * 1000             
        
        dist_me_rat = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
        dist_opp_rat = abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry)
        
        rat_score += expected_rat_value / (dist_me_rat + 1.0)
        rat_score -= expected_rat_value / (dist_opp_rat + 1.0) 
        
        """
    score = point_diff * 1000 + reachability_diff * 100 # + rat_score * 0

    
    #print("point_diff: ", point_diff , "reach:", reachability_diff, "terr", territory_diff, "rat:", rat_score)
            
    return score