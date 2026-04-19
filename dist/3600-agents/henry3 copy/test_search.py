import numpy as np
from game.board import Board
from game.enums import Cell, Direction
from game.move import Move
from expectiminimax import get_best_move
from heuristic import evaluate_board

def create_board_from_ascii(layout: str, player_points: int = 0, opp_points: int = 0) -> Board:
    """
    Parses a visual ASCII layout into a fully functioning Board object.
    Key:
      . = Space
      # = Blocked
      P = Primed
      C = Carpet
      A = Player A (Your Worker)
      B = Player B (Opponent Worker)
    """
    board = Board(build_history=False)
    lines = [line.strip() for line in layout.strip().split('\n') if line.strip()]
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line.split()):
            loc = (x, y)
            if char == '#':
                board.set_cell(loc, Cell.BLOCKED)
            elif char == 'P':
                board.set_cell(loc, Cell.PRIMED)
            elif char == 'C':
                board.set_cell(loc, Cell.CARPET)
            elif char == 'A':
                board.player_worker.position = loc
                board.set_cell(loc, Cell.SPACE)
            elif char == 'B':
                board.opponent_worker.position = loc
                board.set_cell(loc, Cell.SPACE)
            else:
                board.set_cell(loc, Cell.SPACE)
                
    if player_points > 0:
        board.player_worker.increment_points(player_points)
    if opp_points > 0:
        board.opponent_worker.increment_points(opp_points)
        
    return board

def test_full_search():
    # Notice I added spaces between characters so it's easier to read/edit
    test_layout = """
    # . . . . . . #
    . . . B . . . .
    . . P P . . . .
    . . P A P . . .
    . . P . P . . .
    . . . . P . . .
    # . . . . . . #
    . . . . . . . .
    """
    
    print("Building Board State...")
    board = create_board_from_ascii(test_layout, player_points=5, opp_points=5)
    
    # --- 1. Setup Dummy HMM Data ---
    # To isolate the carpeting logic, we assume we have no idea where the rat is.
    # The matrix is 64x64[cite: 22].
    num_cells = 64
    dummy_rat_belief = np.ones(num_cells, dtype=np.float64) / num_cells
    dummy_respawn_belief = np.ones(num_cells, dtype=np.float64) / num_cells
    
    # An identity matrix means the rat never moves (simplest valid transition matrix)
    dummy_hmm_trans = np.eye(num_cells, dtype=np.float64)

    # --- 2. Run Full Search ---
    print("\nRunning Expectiminimax Search (Max Time: 4.0s)...")
    
    best_move = get_best_move(
        board=board,
        max_time=4.0,
        is_player_a=True,
        rat_belief=dummy_rat_belief,
        respawn_belief=dummy_respawn_belief,
        hmm_trans=dummy_hmm_trans
    )
    
    print("\n==================================")
    print(f"🥇 BOT CHOSE: {best_move}")
    print("==================================")

if __name__ == "__main__":
    test_full_search()