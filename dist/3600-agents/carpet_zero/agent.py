import os
import sys
import torch
import numpy as np
from typing import Callable, Tuple

# 1. Path fix for the 'game' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Standard game imports
from game.board import Board
from game.enums import Noise
from game.move import Move

# --- POLYGLOT IMPORT FIX ---
# This allows the bot to work inside the tournament engine AND as a standalone script.
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

class PlayerAgent:
    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CarpetZeroNet()
        
        model_path = "testbest_model.pth"
        if os.path.exists(model_path):
            print("loaded")
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.to(self.device)
        self.model.eval()

        self.serializer = StateSerializer()
        self.hmm = RatHMM(transition_matrix)
        
        self.mcts = AlphaZeroMCTS(
            model=self.model, 
            serializer=self.serializer, 
            num_simulations=150, # This will now be overwritten dynamically every turn
            temperature=0.0 
        )
        
        self.hmm_turns_simulated = 0
        
        # --- HARDWARE BENCHMARK ---
        # Replace this with the result from tune_time.py
        # E.g., if 100 sims takes 1.5 seconds, sims_per_second = 66
        self.sims_per_second = 1500.0

    def commentate(self):
        return "Decoupled AlphaZero initialized. Managing clock dynamically."

    def play(
        self,
        board: Board,
        sensor_data: Tuple[Noise, int],
        time_left: Callable,
    ) -> Move:
        
        # 1. HMM Reset Logic
        opponent_caught_rat = board.opponent_search[1]
        we_caught_rat = board.player_search[1]
        
        if opponent_caught_rat or we_caught_rat:
            self.hmm.reset()
            self.hmm_turns_simulated = board.turn_count 

        # 2. HMM Sync and Update
        turns_to_simulate = board.turn_count - self.hmm_turns_simulated
        for _ in range(turns_to_simulate + 1):
            self.hmm.predict()
        
        self.hmm.update(board, sensor_data, board.player_worker.get_location())
        self.hmm_turns_simulated = board.turn_count + 1

        # 3. HYBRID OVERRIDE LOGIC
        belief_array = self.hmm.get_belief_array()
        max_prob = np.max(belief_array)
        
        if max_prob >= 0.80:
            best_idx = np.argmax(belief_array)
            y = best_idx // 8
            x = best_idx % 8
            return Move.search((x, y))

        # 4. DYNAMIC TIME MANAGEMENT
        # You have 4 minutes total[cite: 5]. Calculate safe budget for this turn.
        my_time = time_left()
        my_turns = board.player_worker.turns_left
        
        # Target time per turn, saving 15% as an emergency buffer
        time_budget_for_turn = (my_time / max(1, my_turns)) * 0.85
        
        # Calculate how many simulations we can afford
        dynamic_sims = int(time_budget_for_turn * self.sims_per_second)
        
        # Cap to a reasonable min/max to prevent anomalous spikes
        self.mcts.num_simulations = max(50, min(dynamic_sims, 800))

        # 5. Standard MCTS Execution
        best_move, _ = self.mcts.search(board, belief_array)

        return best_move