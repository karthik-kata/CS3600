import os
import pathlib
import sys
import time
import multiprocessing
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

from board_utils import get_history_json
from gameplay import play_game
from game.enums import ResultArbiter

def run_single_match(player_a_name, player_b_name, match_index):
    """Worker function to run a single match without terminal output."""
    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    # IMPORTANT: display_game=False so the terminal doesn't turn into spaghetti
    final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b = play_game(
        play_directory,
        play_directory,
        player_a_name,
        player_b_name,
        display_game=False, 
        delay=0.0,
        clear_screen=False,
        record=True,
        limit_resources=False,
    )

    # Save history with a UUID to prevent race conditions when 8 games finish at once
    records_dir = os.path.join(play_directory, "matches")
    os.makedirs(records_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:6]
    out_file = f"{player_a_name}_{player_b_name}_run{match_index}_{unique_id}.json"
    out_path = os.path.join(records_dir, out_file)

    with open(out_path, "w") as fp:
        fp.write(get_history_json(final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b))

    return match_index, final_board.get_winner().name, final_board.get_win_reason().name, final_board.turn_count

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <player_a_name> <player_b_name> [num_matches]")
        sys.exit(1)

    player_a_name = sys.argv[1]
    player_b_name = sys.argv[2]
    num_matches = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"Starting {num_matches} parallel matches between {player_a_name} and {player_b_name}...")
    start_time = time.perf_counter()

    # Track stats
    results = {"A_WINS": 0, "B_WINS": 0, "TIES": 0}

    # Run the pool!
    # max_workers=None will default to the number of CPU cores on your machine.
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all matches to the pool
        futures = {
            executor.submit(run_single_match, player_a_name, player_b_name, i): i 
            for i in range(num_matches)
        }

        # Process them as they finish
        for future in as_completed(futures):
            match_index = futures[future]
            try:
                idx, winner_name, reason, turns = future.result()
                print(f"Match {idx:02d} Finished | Winner: {winner_name:<10} | Reason: {reason:<12} | Turns: {turns}")
                
                if winner_name == "PLAYER_A":
                    results["A_WINS"] += 1
                elif winner_name == "PLAYER_B":
                    results["B_WINS"] += 1
                else:
                    results["TIES"] += 1
                    
            except Exception as e:
                print(f"Match {match_index} crashed: {e}")

    total_time = time.perf_counter() - start_time
    print("\n" + "="*40)
    print("MATCH BATCH COMPLETE")
    print("="*40)
    print(f"Total Time:  {total_time:.2f} seconds")
    print(f"{player_a_name} (A) Wins: {results['A_WINS']}")
    print(f"{player_b_name} (B) Wins: {results['B_WINS']}")
    print(f"Ties:            {results['TIES']}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()