import sys
import os
import json
import torch
import numpy as np
from typing import List, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
engine_path = os.path.join(root_dir, "engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

# 2. Now we can safely import from game
from game.board import Board
from game.enums import Cell, Direction

# 3. Local imports
try:
    from .serializer import StateSerializer
except ImportError:
    from serializer import StateSerializer


def _get_direction(prev_pos: Tuple[int, int], curr_pos: Tuple[int, int], is_player_a: bool) -> Direction:
    dx = curr_pos[1] - prev_pos[1]
    dy = curr_pos[0] - prev_pos[0]
        
    # 2. Coordinate System mapping:
    # If your game's (0,0) is at the TOP-LEFT, moving UP means Y decreases (dy < 0).
    # If your game's (0,0) is at the BOTTOM-LEFT, moving UP means Y increases (dy > 0).
    # (Assuming standard top-left here, swap the UP/DOWN signs if your board is bottom-left)
    if dy < 0: return Direction.UP
    if dy > 0: return Direction.DOWN
    if dx < 0: return Direction.LEFT
    if dx > 0: return Direction.RIGHT
    
    return Direction.UP # Fallback


def parse_match_json(json_path: str) -> List[Tuple]:
    """
    Parses a single match.json file and reconstructs the game state step-by-step
    to generate AlphaZero training data.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    serializer = StateSerializer()
    board = Board(time_to_play=240, build_history=False)
    
    # 1. Apply Initial Blockers
    for x, y in data["blocked_positions"]:
        board.set_cell((x, y), Cell.BLOCKED)

    training_data = []
    
    # Pre-calculate final points for the blended value target
    final_points_a = data["a_points"][-1]
    final_points_b = data["b_points"][-1]

    # Iterate through every turn in the game (Index 0 is the spawn state)
    num_turns = len(data["left_behind"])
    for i in range(1, num_turns):
        
        is_player_a_turn = (i % 2 != 0)
        
        # --- A. RECONSTRUCT THE STATE BEFORE THE MOVE ---
        board.player_worker.position = tuple(data["a_pos"][i-1])
        board.player_worker.points = data["a_points"][i-1]
        board.player_worker.time_left = data["a_time_left"][i-1]
        board.player_worker.turns_left = data["a_turns_left"][i-1]
        
        board.opponent_worker.position = tuple(data["b_pos"][i-1])
        board.opponent_worker.points = data["b_points"][i-1]
        board.opponent_worker.time_left = data["b_time_left"][i-1]
        board.opponent_worker.turns_left = data["b_turns_left"][i-1]
        
        # Orient perspective correctly for the neural network
        if not is_player_a_turn:
            board.reverse_perspective()

        # --- B. FAKE THE HMM BELIEF ---
        # We give the network a 90% accurate belief of where the rat actually was
        belief = np.ones(64, dtype=np.float32) / 64.0
        belief_array = belief.reshape((8, 8))

        # --- C. DEDUCE THE TARGET POLICY (0-35 Index) ---
        move_type_str = data["left_behind"][i]
        
        prev_pos = data["a_pos"][i-1] if is_player_a_turn else data["b_pos"][i-1]
        curr_pos = data["a_pos"][i] if is_player_a_turn else data["b_pos"][i]
        
        action_idx = -1
        
        if move_type_str in ["plain", "prime", "carpet"]:
            # NOW PASSING THE PERSPECTIVE BOOLEAN
            direction = _get_direction(prev_pos, curr_pos, is_player_a_turn) 
            
            if move_type_str == "plain":
                action_idx = int(direction)
            elif move_type_str == "prime":
                action_idx = 4 + int(direction)
            elif move_type_str == "carpet":
                roll_length = max(abs(curr_pos[0] - prev_pos[0]), abs(curr_pos[1] - prev_pos[1]))
                action_idx = 8 + (int(direction) * 7) + (roll_length - 1)
                
        # --- D. GENERATE TRAINING TUPLE ---
        # We only train the network on spatial moves (skip search turns)
        if action_idx != -1:
            # 1. Serialize the board state
            spatial, scalar = serializer.serialize_single(board, belief_array)
            
            # 2. Create One-Hot Policy Target
            target_policy = np.zeros(36, dtype=np.float32)
            target_policy[action_idx] = 1.0
            
            # 3. Calculate Blended Value Target (Win + Margin)
            point_diff = (final_points_a - final_points_b) if is_player_a_turn else (final_points_b - final_points_a)
            win_signal = 1.0 if point_diff > 0 else (-1.0 if point_diff < 0 else 0.0)
            margin_signal = float(np.tanh(point_diff / 20.0))
            
            # The raw final value
            base_value = 0.5 * win_signal + 0.5 * margin_signal
            
            # --- NEW: TIME DISCOUNTING ---
            # Calculate how many turns away the end of the game was
            turns_remaining = num_turns - i 
            
            # Gamma = 0.98 means the signal decays by 2% for every turn we go back in time.
            # A base_value of -0.84 at the end of the game will decay to around -0.45 on Turn 1.
            gamma = 0.98 
            target_value = base_value * (gamma ** turns_remaining)
            # -----------------------------
            
            training_data.append((spatial, scalar, target_policy, target_value))

        # --- E. ADVANCE THE BOARD FOR THE NEXT LOOP ---
        # Revert perspective back to absolute A/B for updating the bitmasks
        if not is_player_a_turn:
            board.reverse_perspective()
            
        if move_type_str == "prime":
            board.set_cell(tuple(prev_pos), Cell.PRIMED)
        elif move_type_str == "carpet":
            for cx, cy in data["new_carpets"][i]:
                board.set_cell((cx, cy), Cell.CARPET)

    return training_data

if __name__ == "__main__":
    # Point this to the match.json you just tested
    test_json = "/Users/karthik/GaTech_Spring_2026/CS3600/CS3600/dist/3600-agents/matches/match.json" 
    
    if os.path.exists(test_json):
        print(f"Parsing {test_json}...")
        data = parse_match_json(test_json)
        print(f"Successfully generated {len(data)} training samples!")
        
        # --- VERIFICATION BLOCK ---
        # Let's peek at the 10th move of the game
        spatial, scalar, policy, value = data[10] 
        print("\n--- DATA VERIFICATION (Turn 10) ---")
        
        # 1. Verify the Value
        print(f"Target Value: {value:.4f}")
        
        # 2. Verify the Policy (Reverse mapping the one-hot array back to a Move)
        try:
            from model import index_to_move
        except ImportError:
            from carpet_zero.model import index_to_move
            
        print("\n--- ALL EXTRACTED MOVES ---")
        # Loop through every extracted turn
        for i, (spatial, scalar, policy, value) in enumerate(data):
            # 1. Reverse map the one-hot policy array back to a Move
            action_idx = int(np.argmax(policy))
            intended_move = index_to_move(action_idx)
            
            # 2. Print it out nicely
            print(f"Extracted Turn {i + 1}: {intended_move} | Value Target: {value:.4f}")
        
    else:
        print(f"Could not find test file at {test_json}.")