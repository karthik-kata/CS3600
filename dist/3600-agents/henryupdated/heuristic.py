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

def evaluate_board(board: Board, rat_belief: np.ndarray = None) -> float:
    """
    Scores the board state from board.player_worker's perspective.
    board.player_worker is always "us" after reverse_perspective() is applied
    correctly in the search tree — no is_player_a flag needed.
    Returns positive if the state favors us, negative if it favors the opponent.
    """
    my_worker = board.player_worker
    opp_worker = board.opponent_worker
        
    # 1. Point Differential (Primary Objective)
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
                my_potential += attr / (dist_me + 1.0)
                opp_potential += attr / (dist_opp + 1.0)
                
    score += (my_potential - opp_potential) * 3.5
    
    # 3. Rat Hunting Potential
    # Use expected value framing: EV(search at best cell) = 6p - 2
    # Also reward proximity to high-probability rat cells (weighted distance).
    if rat_belief is not None:
        best_idx = int(np.argmax(rat_belief))
        p = rat_belief[best_idx]
        search_ev = p * 4.0 - (1.0 - p) * 2.0  # = 6p - 2

        if search_ev > 0:
            score += search_ev * 50.0

        # Weighted distance to rat belief mass — reward being close to where rat likely is
        weighted_dist_me = 0.0
        weighted_dist_opp = 0.0
        for i in range(len(rat_belief)):
            pb = rat_belief[i]
            if pb > 0.005:
                rx, ry = i % BOARD_SIZE, i // BOARD_SIZE
                weighted_dist_me  += pb * (abs(my_loc[0]  - rx) + abs(my_loc[1]  - ry))
                weighted_dist_opp += pb * (abs(opp_loc[0] - rx) + abs(opp_loc[1] - ry))

        # Being closer to the rat than the opponent is good
        score += (weighted_dist_opp - weighted_dist_me) * 10.0
            
    return float(score)