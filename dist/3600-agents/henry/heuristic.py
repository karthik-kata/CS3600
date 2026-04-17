import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

import numpy as np
from game.board import Board
from game.enums import BOARD_SIZE

def _compute_territory_potential(board: Board, px: int, py: int, ox: int, oy: int) -> float:
    """
    Vectorized implementation of the base attractiveness and adjacency weighting,
    evaluated dynamically via a continuous distance gradient.
    """
    # 1. Extract and binarize the board masks
    shifts = np.arange(64, dtype=np.uint64)
    
    def to_grid(mask):
        return ((np.uint64(mask) >> shifts) & np.uint64(1)).reshape((BOARD_SIZE, BOARD_SIZE)).astype(bool)

    primed = to_grid(board._primed_mask)
    space = to_grid(board._space_mask)
    carpet = to_grid(board._carpet_mask)
    
    # 2. Assign Base Weights
    attr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    attr[primed] = 2.0
    attr[space] = 1.0
    attr[carpet] = 0.1
    
    # 3. Calculate Adjacency Boost (0.6 per PRIMED neighbor)
    neighbors = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    neighbors[:, :-1] += primed[:, 1:]  # right
    neighbors[:, 1:] += primed[:, :-1]  # left
    neighbors[:-1, :] += primed[1:, :]  # down
    neighbors[1:, :] += primed[:-1, :]  # up
    
    # Apply the boost only to cells that have a base attractiveness > 0
    valid_mask = attr > 0
    boosted_attr = np.copy(attr)
    boosted_attr[valid_mask] += (neighbors[valid_mask] * 0.6)
    
    # 4. Continuous Distance Weighting
    y_coords, x_coords = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    dist_p = np.abs(x_coords - px) + np.abs(y_coords - py)
    dist_o = np.abs(x_coords - ox) + np.abs(y_coords - oy)
    
    # Weighting: Closer cells exert a stronger gravitational pull on the score
    my_potential = np.sum(boosted_attr / (dist_p + 1.0))
    opp_potential = np.sum(boosted_attr / (dist_o + 1.0))
    
    return float(my_potential - opp_potential)

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
    score = (my_worker.get_points() - opp_worker.get_points()) * 100
    
    
    # 2. Vectorized Board Potential
    buildable_mask = board._space_mask | board._primed_mask
    
    reachability_diff = _compute_reachability(
        buildable_mask, 
        my_loc[0], my_loc[1], 
        opp_loc[0], opp_loc[1]
    )
    
    score += reachability_diff * 10

    
    territory_diff = _compute_territory_potential(
        board, my_loc[0], my_loc[1], opp_loc[0], opp_loc[1]
    )
        
    score += territory_diff * 2
    
    
    
    # 3. Rat Hunting Potential
    if rat_belief is not None:
        best_rat_idx = int(np.argmax(rat_belief))
        rx = best_rat_idx % BOARD_SIZE
        ry = best_rat_idx // BOARD_SIZE
        max_prob = rat_belief[best_rat_idx]
        
        if max_prob > 0.15:
            dist_me_rat = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
            dist_opp_rat = abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry)
            
            expected_rat_value = max_prob * 200             
            score += expected_rat_value / (dist_me_rat + 1.0)
            score -= expected_rat_value / (dist_opp_rat + 1.0) 
            
    return score