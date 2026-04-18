import os
import random
import pathlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from gameplay import play_game

# ---------------- CONFIGURATION ----------------
GAMES_PER_WEIGHT = 8
NUM_CONFIGS_TO_TEST = 20
YOUR_BOT_NAME = "henry_train"  # Change to your agent's folder name
OPPONENT_NAME = "henry_original"
# -----------------------------------------------

def evaluate_single_match(weights):
    """
    Runs a single game in an isolated process with specific environment variables.
    """
    # Inject weights into the environment for this specific process
    os.environ["W_POINT"] = str(weights["W_POINT"])
    os.environ["W_REACH"] = str(weights["W_REACH"])
    os.environ["W_TERR"] = str(weights["W_TERR"])
    os.environ["W_RAT"] = str(weights["W_RAT"])

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    # Run the game silently
    final_board, _, _, _, _, _ = play_game(
        play_directory, play_directory,
        YOUR_BOT_NAME, OPPONENT_NAME,
        display_game=False,  # Turn off rendering for speed
        delay=0.0, clear_screen=False, record=False, limit_resources=False
    )

    # Calculate fitness (Point Differential)
    my_points = final_board.player_worker.get_points()
    opp_points = final_board.opponent_worker.get_points()
    
    # Positive diff means your bot won, negative means opponent won
    return my_points - opp_points 

def test_configuration(weights):
    """
    Runs GAMES_PER_WEIGHT matches for a given weight configuration in parallel.
    """
    print(f"Testing weights: {weights}...")
    
    # Run games in parallel to save time
    with ProcessPoolExecutor(max_workers=GAMES_PER_WEIGHT) as executor:
        # Submit the same weights for 8 independent matches
        results = list(executor.map(evaluate_single_match, [weights] * GAMES_PER_WEIGHT))
    
    avg_point_diff = sum(results) / len(results)
    win_rate = sum(1 for diff in results if diff > 0) / len(results)
    
    print(f"--> Avg Point Diff: {avg_point_diff:.2f} | Win Rate against {OPPONENT_NAME}: {win_rate*100:.0f}%")
    return avg_point_diff

def main():
    best_weights = None
    best_score = float('-inf')

    for i in range(NUM_CONFIGS_TO_TEST):
        print(f"\n--- Configuration {i+1}/{NUM_CONFIGS_TO_TEST} ---")
        
        # Generate random weights within reasonable bounds
        current_weights = {
            "W_POINT": random.uniform(500, 5000),   # Lowered from 10000 to balance
            "W_REACH": random.uniform(50, 500),
            "W_TERR": random.uniform(5, 50),
            "W_RAT": random.uniform(10, 200)        # Boosted to encourage rat hunting
        }

        score = test_configuration(current_weights)

        if score > best_score:
            best_score = score
            best_weights = current_weights
            print(f"🏆 NEW BEST CONFIG FOUND! Score: {best_score:.2f}")

    print("\n==================================================")
    print("FINISHED TRAINING. BEST WEIGHTS:")
    for k, v in best_weights.items():
        print(f"{k}: {v:.2f}")
    print(f"Expected Point Differential vs {OPPONENT_NAME}: {best_score:.2f}")
    print("==================================================")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()