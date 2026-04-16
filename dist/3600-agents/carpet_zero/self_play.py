import sys
import os

# Ensure Python can find the 'game' module from the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
engine_path = os.path.join(root_dir, "engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from game.board import Board
from game.enums import Result, WinReason, Cell, MoveType
from game.move import Move
from game.rat import Rat
from board_utils import generate_spawns

# Local imports
try:
    from .model import CarpetZeroNet
    from .serializer import StateSerializer
    from .hmm import RatHMM
    from .mcts import AlphaZeroMCTS
except (ImportError, ValueError):
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

# --- 2. Random Transition Matrix Generator ---
def generate_random_transition_matrix() -> np.ndarray:
    """
    Generates a valid 64x64 transition matrix where the rat can only move to 
    adjacent cardinal squares or stay in place, and probabilities sum to 1.0.
    """
    T = np.zeros((64, 64), dtype=np.float32)
    
    for i in range(64):
        x = i % 8
        y = i // 8
        
        valid_moves = [i] # Can always stay in place
        if y > 0: valid_moves.append(i - 8) # Up
        if y < 7: valid_moves.append(i + 8) # Down
        if x > 0: valid_moves.append(i - 1) # Left
        if x < 7: valid_moves.append(i + 1) # Right
        
        # Generate random weights for valid moves and normalize to sum to 1.0
        weights = np.random.uniform(0.1, 1.0, size=len(valid_moves))
        probs = weights / np.sum(weights)
        
        for move_idx, prob in zip(valid_moves, probs):
            T[i, move_idx] = prob
            
    return T

# --- 3. Headless Game Simulation ---
def play_single_game(process_id: int, model_weights_path: str) -> List[Tuple]:
    """
    Runs a single self-play game entirely in memory. Optimized for speed.
    """
    device = torch.device("cpu") 
    
    model = CarpetZeroNet()
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    serializer = StateSerializer()
    
    # Generate a unique transition matrix for this specific game
    T_matrix = generate_random_transition_matrix()
    
    hmm_a = RatHMM(T_matrix)
    hmm_b = RatHMM(T_matrix)
    
    mcts_a = AlphaZeroMCTS(model, serializer, num_simulations=100, c_puct=1.25, temperature=1.0)
    mcts_b = AlphaZeroMCTS(model, serializer, num_simulations=100, c_puct=1.25, temperature=1.0)

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

    # Both players spawn in mirrored positions within the center 4x4 [cite: 16]
    spawn_a, spawn_b = generate_spawns(board)
    board.player_worker.position = spawn_a
    board.opponent_worker.position = spawn_b
    
    # Rat spawns at (0,0) and takes 1000 moves before the game starts [cite: 28]
    rat.spawn()

    game_history = []
    turns_since_a_hmm_sync = 0
    turns_since_b_hmm_sync = 0

    while not board.is_game_over():
        # The rat moves and makes noise before each turn [cite: 48]
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
        hmm.update(board, sensor_data, board.player_worker.get_location())
        
        if current_player_is_a:
            turns_since_a_hmm_sync = 0
            turns_since_b_hmm_sync += 1
        else:
            turns_since_b_hmm_sync = 0
            turns_since_a_hmm_sync += 1
        
        if board.turn_count < 30:
            mcts.temperature = 1.0
        else:
            mcts.temperature = 0.01

        belief = hmm.get_belief_array()
        if np.max(belief) >= 0.85:
            best_idx = np.argmax(belief)
            move = Move.search((best_idx % 8, best_idx // 8))
        else:
            # Pass the 9th channel belief state to the MCTS and Serializer
            move, policy = mcts.search(board, belief)
            spatial, scalar = serializer.serialize_single(board, belief)
            game_history.append((spatial, scalar, policy, current_player_is_a))
            
        board.apply_move(move, timer=0.1, check_ok=False)
        
        if move.move_type == MoveType.SEARCH:
            if move.search_loc == rat.get_position():
                board.player_worker.increment_points(4) # Win 4 points if correct [cite: 47]
                rat.spawn()
                if current_player_is_a:
                    board.player_search = (move.search_loc, True)
                else:
                    board.player_search = (move.search_loc, True)
            else:
                board.player_worker.decrement_points(2) # Lose 2 points if wrong [cite: 47]
                if current_player_is_a:
                    board.player_search = (move.search_loc, False)
                else:
                    board.player_search = (move.search_loc, False)

        if not board.is_game_over():
            board.reverse_perspective()

    if board.is_player_a_turn:
        final_points_a = board.player_worker.get_points()
        final_points_b = board.opponent_worker.get_points()
    else:
        final_points_a = board.opponent_worker.get_points()
        final_points_b = board.player_worker.get_points()

    training_data = []
    
    for spatial, scalar, policy, is_player_a in game_history:
        # 1. Calculate the final point differential from the perspective 
        # of the player who made this specific move.
        if is_player_a:
            point_diff = final_points_a - final_points_b
        else:
            point_diff = final_points_b - final_points_a
            
        # 2. Binary Win/Loss Signal
        # Maintains the fundamental drive to win the game.
        if point_diff > 0:
            win_signal = 1.0
        elif point_diff < 0:
            win_signal = -1.0
        else:
            win_signal = 0.0
            
        # 3. Margin of Victory Signal
        # We use np.tanh to bound the differential between [-1.0, 1.0].
        # A divisor of 20.0 means a 21-point lead (equivalent to a length-7 carpet) 
        # will yield a strong margin signal of ~0.78.
        margin_signal = float(np.tanh(point_diff / 20.0))
        
        # 4. Blended Value Target
        # 50% weight on actually winning, 50% weight on maximizing the point gap.
        # This prevents the bot from playing passively when it has a tiny lead, 
        # while strongly punishing it for missing the rat and losing 2 points.
        value = 0.5 * win_signal + 0.5 * margin_signal
        
        training_data.append((spatial, scalar, policy, value))

    return training_data

# --- 4. Parallel Training Pipeline ---
def train_alphazero(num_iterations: int = 50, games_per_iter: int = 100, epochs: int = 3, batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarpetZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print("model loaded")
        model.load_state_dict(torch.load(model_path))

    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Starting AlphaZero training using {num_cores} parallel workers...")

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        pool = mp.Pool(processes=num_cores)
        jobs = []
        for i in range(games_per_iter):
            jobs.append(pool.apply_async(play_single_game, args=(i, model_path)))
            
        pool.close()
        pool.join()
        
        iteration_data = []
        for job in jobs:
            iteration_data.extend(job.get())
            
        print(f"Generated {len(iteration_data)} board states from {games_per_iter} games.")
        
        dataset = AlphaZeroDataset(iteration_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for spatial, scalar, target_policy, target_value in dataloader:
                spatial = spatial.to(device)
                scalar = scalar.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                
                optimizer.zero_grad()
                
                pred_policy_logits, pred_value = model(spatial, scalar)
                
                value_loss = F.mse_loss(pred_value, target_value)
                policy_loss = F.cross_entropy(pred_policy_logits, target_policy)
                
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{epochs} - Average Loss: {total_loss / max(1, len(dataloader)):.4f}")
            
        torch.save(model.state_dict(), model_path)
        print("Updated best_model.pth")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train_alphazero()