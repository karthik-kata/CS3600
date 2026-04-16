import numpy as np
from game.board import Board
from game.enums import Cell, BOARD_SIZE

def calculate_cell_attractiveness(board: Board) -> np.ndarray:
    """
    Evaluates the base potential of every cell on the board using specific discrete weights.
    Identifies high-value clusters (e.g., contiguous primed cells for carpeting).
    """
    attractiveness = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    
    # Base weighting logic
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            cell_type = board.get_cell((x, y))
            if cell_type == Cell.PRIMED:
                attractiveness[y, x] = 2.0  # High value: ready for carpet
            elif cell_type == Cell.SPACE:
                attractiveness[y, x] = 1.0  # Moderate value: can be primed
            elif cell_type == Cell.CARPET:
                attractiveness[y, x] = 0.1  # Low value: already claimed, but passable
            elif cell_type == Cell.BLOCKED:
                attractiveness[y, x] = 0.0  # Zero value
                
    # Adjacency weighting to proxy high-yield regions
    boosted_attractiveness = np.copy(attractiveness)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if attractiveness[y, x] > 0:
                neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if board.get_cell((nx, ny)) == Cell.PRIMED:
                            neighbors += 1
                # Synergistic boost for contiguous primed lines (carpet potential)
                boosted_attractiveness[y, x] += (neighbors * 0.6)
                
    return boosted_attractiveness

def evaluate_board(board: Board, is_player_a: bool, rat_belief: np.ndarray = None) -> float:
    """
    Scores the board state.
    Returns a positive float if the state favors the requesting player, negative if it favors the opponent.
    """
    # Identify which worker object belongs to the evaluating player
    if board.is_player_a_turn == is_player_a:
        my_worker = board.player_worker
        opp_worker = board.opponent_worker
    else:
        my_worker = board.opponent_worker
        opp_worker = board.player_worker
        
    # 1. Point Differential (Primary Objective)
    # Scaled heavily so actual points outweigh potential
    score = (my_worker.get_points() - opp_worker.get_points()) * 100.0
    
    # 2. Board Potential (Attractiveness weighted by distance)
    attractiveness_matrix = calculate_cell_attractiveness(board)
    
    my_loc = my_worker.get_location()
    opp_loc = opp_worker.get_location()
    
    my_potential = 0.0
    opp_potential = 0.0
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            attr = attractiveness_matrix[y, x]
            if attr > 0:
                dist_me = abs(my_loc[0] - x) + abs(my_loc[1] - y)
                dist_opp = abs(opp_loc[0] - x) + abs(opp_loc[1] - y)
                
                # Weighting: Closer cells are more achievable. Epsilon of 1.0 prevents division by zero.
                my_potential += attr / (dist_me + 1.0)
                opp_potential += attr / (dist_opp + 1.0)
                
    # Add net potential to the score
    score += (my_potential - opp_potential) * 3.5
    
    # 3. Rat Hunting Potential
    if rat_belief is not None:
        best_rat_idx = int(np.argmax(rat_belief))
        rx = best_rat_idx % BOARD_SIZE
        ry = best_rat_idx // BOARD_SIZE
        max_prob = rat_belief[best_rat_idx]
        
        # Only factor rat distance if confidence is reasonably high
        if max_prob > 0.15:
            dist_me_rat = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
            dist_opp_rat = abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry)
            
            # The 4-point rat bonus is equivalent to 400 heuristic units
            expected_rat_value = max_prob * 350.0 
            
            score += expected_rat_value / (dist_me_rat + 1.0)
            score -= expected_rat_value / (dist_opp_rat + 1.0)
            
    return float(score)