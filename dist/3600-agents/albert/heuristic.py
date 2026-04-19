from game.board import Board
from game.enums import Cell, CARPET_POINTS_TABLE
import numpy as np

def _get_straight_prime_length(board: Board, start_loc: tuple, dx: int, dy: int) -> int:
    """Helper to find how many primed tiles extend in a specific direction."""
    length = 0
    curr_loc = (start_loc[0] + dx, start_loc[1] + dy)
    
    # Cap at 7 since that's the max carpet roll length
    while length < 7 and board.is_valid_cell(curr_loc) and board.get_cell(curr_loc) == Cell.PRIMED:
        length += 1
        curr_loc = (curr_loc[0] + dx, curr_loc[1] + dy)
        
    return length

def _evaluate_L_shaped_carpet(board: Board, worker_loc: tuple) -> float:
    """
    Calculates the exact maximum points from rolling up to TWO straight lines 
    (an L-shape combo) starting from the worker.
    """
    max_points = 0.0
    
    # Map directions to their orthogonal (perpendicular) options
    directions = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
    }
    orthogonals = {
        'UP': ['LEFT', 'RIGHT'],
        'DOWN': ['LEFT', 'RIGHT'],
        'LEFT': ['UP', 'DOWN'],
        'RIGHT': ['UP', 'DOWN']
    }
    
    for d1_name, (dx1, dy1) in directions.items():
        # Find how far we can roll in the primary direction
        max_l1 = _get_straight_prime_length(board, worker_loc, dx1, dy1)
        
        # We must check EVERY stopping point along this line, because 
        # stopping early might align us with a massive perpendicular branch!
        for l1 in range(1, max_l1 + 1):
            score1 = CARPET_POINTS_TABLE.get(l1, 0)
            
            # Where does the worker land after this first roll?
            corner_loc = (worker_loc[0] + dx1 * l1, worker_loc[1] + dy1 * l1)
            
            # Check for a second roll perpendicular to the first
            best_l2_score = 0
            for d2_name in orthogonals[d1_name]:
                dx2, dy2 = directions[d2_name]
                l2 = _get_straight_prime_length(board, corner_loc, dx2, dy2)
                if l2 > 0:
                    best_l2_score = max(best_l2_score, CARPET_POINTS_TABLE.get(l2, 0))
            
            total_combo_score = score1 + best_l2_score
            if total_combo_score > max_points:
                max_points = total_combo_score
                
    return float(max_points)

def evaluate_board(board: Board, is_player_a: bool, rat_belief=None) -> float:
    """
    Evaluates the board based purely on acquired points and the exact 
    point value of L-shaped carpet combinations available to the workers.
    """
    if board.is_player_a_turn == is_player_a:
        my_worker = board.player_worker
        opp_worker = board.opponent_worker
    else:
        my_worker = board.opponent_worker
        opp_worker = board.player_worker
        
    my_loc = my_worker.get_location()
    opp_loc = opp_worker.get_location()
        
    # 1. Current Real Points
    my_real_points = my_worker.get_points()
    opp_real_points = opp_worker.get_points()
    
    # 2. Potential Points from L-shaped bends
    my_combo_points = _evaluate_L_shaped_carpet(board, my_loc)
    opp_combo_points = _evaluate_L_shaped_carpet(board, opp_loc)
    
    # We multiply the potential combo points by a slight discount (e.g., 0.9).
    # This prevents the bot from stalling; it ensures that 10 REAL points 
    # will always mathematically beat 10 UNROLLED points, forcing it to cash out.
    my_total_value = my_real_points + (my_combo_points * 0.4)
    opp_total_value = opp_real_points + (opp_combo_points * 0.4)
    
    point_diff = my_total_value - opp_total_value
        
    # Notice we don't need wild multipliers like 10,000 anymore because 
    # the points are scaled to exact game values!
    score = point_diff
    
    return score