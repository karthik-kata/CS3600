import os
import sys

# 1. Dynamically find the project root and the current directory
# This allows the script to find 'game' and local modules regardless of where it's called from.
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)             # .../carpet_zero
project_root = os.path.abspath(os.path.join(current_dir, "../../")) # .../dist

# 2. Add both to sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Add the engine folder specifically so 'import game' works
engine_path = os.path.join(project_root, "engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

# --- Standard Library and Third Party Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import torch.nn.functional as F

# --- Game Engine Imports ---
# These now work because engine_path is in sys.path
from game.board import Board
from game.enums import Result, WinReason, Cell
from game.rat import Rat
from game.move import Move, MoveType
from engine.board_utils import generate_spawns # Adjusted to include engine prefix if needed

# --- Local Agent Imports ---
# REMOVED the dots (.) to convert these to absolute imports.
# These work because current_dir is in sys.path.
from model import CarpetZeroNet
from serializer import StateSerializer
from hmm import RatHMM
from mcts import AlphaZeroMCTS

# --- 1. Experience Replay Dataset ---
class AlphaZeroDataset(Dataset):
    """Stores game data for PyTorch training."""
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor, np.ndarray, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spatial, scalar, policy, value = self.data[idx]
        return spatial.squeeze(0), scalar.squeeze(0), torch.tensor(policy), torch.tensor([value], dtype=torch.float32)

# --- 2. Headless Game Simulation ---
def play_single_game(process_id: int, model_weights_path: str, T_matrix: np.ndarray) -> List[Tuple]:
    """
    Runs a single self-play game entirely in memory, without the heavy IPC queues 
    of the tournament gameplay.py engine. Optimized for speed.
    """
    device = torch.device("cpu") # For rollout generation, CPU is often faster per-thread than GPU context switching
    
    model = CarpetZeroNet()
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    serializer = StateSerializer()
    
    # We maintain separate HMMs for Player A and Player B to respect hidden information
    hmm_a = RatHMM(T_matrix)
    hmm_b = RatHMM(T_matrix)
    
    # MCTS instances with temperature=1.0 for exploration
    mcts_a = AlphaZeroMCTS(model, serializer, num_simulations=50, temperature=1.0)
    mcts_b = AlphaZeroMCTS(model, serializer, num_simulations=50, temperature=1.0)

    # Initialize headless game [cite: 3-4, 15-16]
    board = Board(time_to_play=240, build_history=False)
    rat = Rat(T_matrix)
    
    # Generate random corner blockers [cite: 15]
    shapes = [(2, 3), (3, 2), (2, 2)]
    for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        w, h = shapes[np.random.choice(len(shapes))]
        for dx in range(w):
            for dy in range(h):
                x = dx if ox == 0 else 7 - dx
                y = dy if oy == 0 else 7 - dy
                board.set_cell((x, y), Cell.BLOCKED)

    spawn_a, spawn_b = generate_spawns(board)
    board.player_worker.position = spawn_a
    board.opponent_worker.position = spawn_b
    rat.spawn() 

    # Data collection buffers
    game_history = []
    
    turns_since_a_hmm_sync = 0
    turns_since_b_hmm_sync = 0

    while not board.is_game_over():
        rat.move() 
        sensor_data = rat.sample(board)
        
        current_player_is_a = board.is_player_a_turn
        hmm = hmm_a if current_player_is_a else hmm_b
        mcts = mcts_a if current_player_is_a else mcts_b
        turns_since_sync = turns_since_a_hmm_sync if current_player_is_a else turns_since_b_hmm_sync

        # HMM Reset if rat was caught [cite: 55]
        if board.opponent_search[1] or board.player_search[1]:
            hmm.reset()
            turns_since_sync = 0

        # Sync HMM
        for _ in range(turns_since_sync + 1):
            hmm.predict()
        hmm.update(board, sensor_data, board.player_worker.get_location())
        
        if current_player_is_a:
            turns_since_a_hmm_sync = 0
            turns_since_b_hmm_sync += 1
        else:
            turns_since_b_hmm_sync = 0
            turns_since_a_hmm_sync += 1

        # Hybrid Override Logic
        belief = hmm.get_belief_array()
        if np.max(belief) >= 0.85:
            best_idx = np.argmax(belief)
            move = Move.search((best_idx % 8, best_idx // 8))
            # We DO NOT record search moves in the training data, as the NN only predicts 36 spatial moves
        else:
            move, policy = mcts.search(board)
            # Record state, policy, and current player id (True=A, False=B)
            spatial, scalar = serializer.serialize_single(board)
            game_history.append((spatial, scalar, policy, current_player_is_a))
            
        # Execute Move
        board.apply_move(move, timer=0.1, check_ok=False)
        
        # Check rat catch mechanics [cite: 43-44, 47]
        if move.move_type == MoveType.SEARCH:
            if move.search_loc == rat.get_position():
                board.player_worker.increment_points(4)
                rat.spawn()
                if current_player_is_a:
                    board.player_search = (move.search_loc, True)
                else:
                    board.player_search = (move.search_loc, True)
            else:
                board.player_worker.decrement_points(2)
                if current_player_is_a:
                    board.player_search = (move.search_loc, False)
                else:
                    board.player_search = (move.search_loc, False)

        if not board.is_game_over():
            board.reverse_perspective()

    # Determine perspectives for value assignment
    winner = board.get_winner()
    training_data = []
    
    for spatial, scalar, policy, is_player_a in game_history:
        value = 0.0
        if winner == Result.TIE:
            value = 0.0
        elif (winner == Result.PLAYER and is_player_a) or (winner == Result.ENEMY and not is_player_a):
            value = 1.0 # The player whose turn it was won
        else:
            value = -1.0 # The player whose turn it was lost
            
        training_data.append((spatial, scalar, policy, value))

    return training_data

# --- 3. Parallel Training Pipeline ---
def train_alphazero(num_iterations: int = 50, games_per_iter: int = 100, epochs: int = 5, batch_size: int = 128):
    """
    Main loop for generating games via multiprocessing and training the network.
    """
    # Create transition matrix dummy (in reality, load from transition_matrices folder)
    # Using uniform distribution for the sake of starting the script
    T_matrix = np.ones((64, 64)) / 64.0 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarpetZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Starting AlphaZero training using {num_cores} parallel workers...")

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # 1. Generate Games using Multiprocessing
        pool = mp.Pool(processes=num_cores)
        jobs = []
        for i in range(games_per_iter):
            jobs.append(pool.apply_async(play_single_game, args=(i, model_path, T_matrix)))
            
        pool.close()
        pool.join()
        
        # 2. Collect Data
        iteration_data = []
        for job in jobs:
            iteration_data.extend(job.get())
            
        print(f"Generated {len(iteration_data)} board states from {games_per_iter} games.")
        
        dataset = AlphaZeroDataset(iteration_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 3. Train Network 
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for spatial, scalar, target_policy, target_value in dataloader:
                spatial = spatial.to(device)
                scalar = scalar.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pred_policy_logits, pred_value = model(spatial, scalar)
                
                # Loss Calculation
                # Value loss: Mean Squared Error (how close did we guess the win/loss?)
                value_loss = F.mse_loss(pred_value, target_value)
                
                # Policy loss: Cross Entropy (how close did our logits match the MCTS probabilities?)
                # CrossEntropyLoss expects logits, so we apply it directly to pred_policy_logits
                policy_loss = F.cross_entropy(pred_policy_logits, target_policy)
                
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{epochs} - Average Loss: {total_loss / len(dataloader):.4f}")
            
        # Save the new "best" model weights
        torch.save(model.state_dict(), model_path)
        print("Updated best_model.pth")

if __name__ == '__main__':
    # Required for safe multiprocessing in Python
    mp.set_start_method('spawn')
    train_alphazero()