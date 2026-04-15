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
def play_single_game(iteration_id: int, model_weights_path: str, T_matrix: np.ndarray) -> List[Tuple]:
    """
    Runs a single self-play game sequentially. 
    Includes print statements for easier debugging of HMM and MCTS logic.
    """
    device = torch.device("cpu") 
    model = CarpetZeroNet()
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    serializer = StateSerializer()
    hmm_a = RatHMM(T_matrix)
    hmm_b = RatHMM(T_matrix)
    
    mcts_a = AlphaZeroMCTS(model, serializer, num_simulations=50, temperature=1.0)
    mcts_b = AlphaZeroMCTS(model, serializer, num_simulations=50, temperature=1.0)

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

        if board.opponent_search[1] or board.player_search[1]:
            hmm.reset()
            turns_since_sync = 0

        for _ in range(turns_since_sync + 1):
            hmm.predict()
        
        # This is where the IndexError likely occurs. 
        # By running sequentially, you will get a clear traceback here.
        hmm.update(board, sensor_data, board.player_worker.get_location())
        
        if current_player_is_a:
            turns_since_a_hmm_sync = 0
            turns_since_b_hmm_sync += 1
        else:
            turns_since_b_hmm_sync = 0
            turns_since_a_hmm_sync += 1

        belief = hmm.get_belief_array()
        if np.max(belief) >= 0.85:
            best_idx = np.argmax(belief)
            move = Move.search((best_idx % 8, best_idx // 8))
        else:
            move, policy = mcts.search(board)
            spatial, scalar = serializer.serialize_single(board)
            game_history.append((spatial, scalar, policy, current_player_is_a))
            
        board.apply_move(move, timer=0.1, check_ok=False)
        
        # Handle rat capture [cite: 43-44, 47]
        if move.move_type == MoveType.SEARCH:
            if move.search_loc == rat.get_position():
                board.player_worker.increment_points(4)
                rat.spawn()
                board.player_search = (move.search_loc, True)
            else:
                board.player_worker.decrement_points(2)
                board.player_search = (move.search_loc, False)

        if not board.is_game_over():
            board.reverse_perspective()

    winner = board.get_winner()
    training_data = []
    for spatial, scalar, policy, is_player_a in game_history:
        value = 0.0
        if winner == Result.TIE:
            value = 0.0
        elif (winner == Result.PLAYER and is_player_a) or (winner == Result.ENEMY and not is_player_a):
            value = 1.0
        else:
            value = -1.0
        training_data.append((spatial, scalar, policy, value))

    return training_data

# --- 3. Sequential Training Pipeline ---
def train_alphazero_sequential(num_iterations: int = 5, games_per_iter: int = 2, epochs: int = 5, batch_size: int = 16):
    """
    Main loop configured for sequential execution to allow for easy debugging.
    """
    # Initialize dummy T matrix [cite: 22-23]
    T_matrix = np.ones((64, 64)) / 64.0 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarpetZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    print("Starting SEQUENTIAL AlphaZero training for debugging...")

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        iteration_data = []
        for i in range(games_per_iter):
            print(f"  Playing game {i+1}/{games_per_iter}...")
            game_results = play_single_game(i, model_path, T_matrix)
            iteration_data.extend(game_results)
            
        print(f"Generated {len(iteration_data)} states. Starting training...")
        
        dataset = AlphaZeroDataset(iteration_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for spatial, scalar, target_policy, target_value in dataloader:
                spatial, scalar = spatial.to(device), scalar.to(device)
                target_policy, target_value = target_policy.to(device), target_value.to(device)
                
                optimizer.zero_grad()
                pred_policy_logits, pred_value = model(spatial, scalar)
                
                loss = F.mse_loss(pred_value, target_value) + F.cross_entropy(pred_policy_logits, target_policy)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"    Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
            
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_alphazero_sequential()