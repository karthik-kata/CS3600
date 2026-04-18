import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

def _compute_territory_potential(board: Board, px: int, py: int, ox: int, oy: int) -> float:
    shifts = np.arange(64, dtype=np.uint64)
    
    def to_grid(mask):
        return ((np.uint64(mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)

    primed = to_grid(board._primed_mask)
    space = to_grid(board._space_mask)
    carpet = to_grid(board._carpet_mask)
    
    # Base Weights
    attr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    attr[primed] = 2.0
    attr[space] = 1.0
    # Carpet is dead terrain, ignore it
    
    y_coords, x_coords = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    dist_p = np.abs(x_coords - px) + np.abs(y_coords - py)
    dist_o = np.abs(x_coords - ox) + np.abs(y_coords - oy)
    
    # Simple, smooth gravity towards buildable terrain. 
    # Let the Reachability function handle the "Line" shapes.
    my_potential = np.sum(attr / (dist_p + 1.0))
    opp_potential = np.sum(attr / (dist_o + 1.0))
    
    return float(my_potential - opp_potential)

def _compute_reachability(space_mask: int, primed_mask: int, px: int, py: int, ox: int, oy: int) -> float:
    """
    Standard NumPy function for fast vectorized reachability evaluation.
    Accounts for the exponential scoring of longer lines and heavily weights primed terrain.
    """
    shifts = np.arange(64, dtype=np.uint64)
    
    # 1. State Binarization via Bitboard Extraction
    space = ((np.uint64(space_mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)
    primed = ((np.uint64(primed_mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)
    buildable = space | primed
    
    # 2. Base Values 
    # Give PRIMED a significantly higher base weight so that lines with more primed tiles scale harder.
    val = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    val[space] = 1.0
    val[primed] = 3.0   #CHANGE THIS FOR
    
    # 3. Vectorized Line Integration
    # Accumulate the total value of contiguous segments in all four directions.
    h_fwd, h_bwd = np.copy(val), np.copy(val)
    v_fwd, v_bwd = np.copy(val), np.copy(val)
    
    # Max line length on an 8x8 board is 8, so 7 shifts guarantee full propagation.
    for _ in range(7):
        h_fwd[:, 1:] = (h_fwd[:, :-1] + val[:, 1:]) * buildable[:, 1:]
        h_bwd[:, :-1] = (h_bwd[:, 1:] + val[:, :-1]) * buildable[:, :-1]
        v_fwd[1:, :] = (v_fwd[:-1, :] + val[1:, :]) * buildable[1:, :]
        v_bwd[:-1, :] = (v_bwd[1:, :] + val[:-1, :]) * buildable[:-1, :]
        
    # 4. Exponential Scaling
    # Summing forward and backward (and subtracting the double-counted center tile) 
    # gives the FULL contiguous segment value to EVERY tile in that segment.
    # Squaring this value creates the exponential reward curve for longer lines.
    h_run_value = (h_fwd + h_bwd - val) ** 2
    v_run_value = (v_fwd + v_bwd - val) ** 2
    
    # A tile's true potential is its best orientation.
    tile_potential = np.maximum(h_run_value, v_run_value)
    
    # 5. Entry Point Extraction
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
    
    # Filter our massive potential matrix so we only get "pulled" toward the ends of lines.
    entry_values = tile_potential * valid_entries
    
    # 6. Graph Reachability Race
    y_coords, x_coords = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    
    # Calculate Manhattan Distances
    dist_p = np.abs(x_coords - px) + np.abs(y_coords - py)
    dist_o = np.abs(x_coords - ox) + np.abs(y_coords - oy)
    
    # NOTE: Changed 0.001 to 1.0. 
    # A denominator of +0.001 causes a 1,000x score explosion if a worker is currently standing on an entry point, 
    # causing erratic heuristic behavior. +1.0 ensures smooth, rational gravity.
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
    
    territory_diff = _compute_territory_potential(
        board, my_loc[0], my_loc[1], opp_loc[0], opp_loc[1]
    )
        
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
        
    
    score = point_diff * 10000 + reachability_diff * 50 + territory_diff * 1 + rat_score * 20

    
    #print("point_diff: ", point_diff , "reach:", reachability_diff, "terr", territory_diff, "rat:", rat_score)
            
    return score