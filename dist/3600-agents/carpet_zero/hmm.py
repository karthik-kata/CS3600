import numpy as np
from typing import Tuple
from game.board import Board
from game.enums import Cell, Noise

class RatHMM:
    """
    Highly optimized Hidden Markov Model for tracking the CS3600 Rat.
    Uses fully vectorized NumPy operations for high-throughput MCTS compatibility.
    """
    def __init__(self, T_matrix: np.ndarray):
        """
        Args:
            T_matrix: The 64x64 transition probability matrix[cite: 22, 23].
        """
        self.T = np.array(T_matrix, dtype=np.float32)
        
        # Precompute the 1000-step headstart matrix[cite: 28, 56].
        # Matrix power is expensive, so we do it exactly once at initialization.
        self.T_1000 = np.linalg.matrix_power(self.T, 1000)
        
        # Precompute the emission probability tables for O(1) vectorized lookups
        self._init_emission_tables()
        
        # Precompute a coordinate grid for fast Manhattan distance calculations
        self.grid_x, self.grid_y = np.meshgrid(np.arange(8), np.arange(8))
        self.grid_x_flat = self.grid_x.flatten()
        self.grid_y_flat = self.grid_y.flatten()
        
        # Set the initial belief distribution
        self.reset()

    def _init_emission_tables(self):
        """
        Precomputes the likelihood tables P(Observation | State) to eliminate 
        branching logic during the MCTS rollout phase.
        """
        # 1. Noise Probabilities: P(Noise | Cell Type) [cite: 50]
        # Shape: (4, 3) -> Rows correspond to Cell enum values, Columns to Noise enum values
        self.P_noise = np.zeros((4, 3), dtype=np.float32)
        self.P_noise[Cell.SPACE]   = [0.15, 0.70, 0.15]
        self.P_noise[Cell.PRIMED]  = [0.80, 0.10, 0.10]
        self.P_noise[Cell.CARPET]  = [0.10, 0.10, 0.80]
        self.P_noise[Cell.BLOCKED] = [0.30, 0.50, 0.20]

        # 2. Distance Probabilities: P(Estimated Distance | Actual Distance) [cite: 51]
        # Max manhattan distance on an 8x8 board is 14. Max estimate is 14 + 2 = 16.
        self.P_dist = np.zeros((15, 17), dtype=np.float32)
        
        for actual_d in range(15):
            # Error distributions [cite: 51]
            offsets = [(-1, 0.12), (0, 0.70), (1, 0.12), (2, 0.06)]
            for offset, prob in offsets:
                est_d = actual_d + offset
                # Constraint: Estimates less than zero are clamped to zero [cite: 54]
                est_d = max(0, est_d)
                self.P_dist[actual_d, est_d] += prob

    def reset(self):
        """
        Resets the HMM when a new game starts or when a rat is caught.
        The rat spawns at (0,0) and immediately takes 1000 steps[cite: 28, 55, 56].
        """
        initial_state = np.zeros(64, dtype=np.float32)
        initial_state[0] = 1.0 # Rat spawns at (0,0) [cite: 28]
        
        # Apply the 1000 step headstart using the precomputed matrix
        self.belief = initial_state @ self.T_1000
        
    def predict(self):
        """
        Propagates the belief forward one time step using the transition matrix.
        Called BEFORE evaluating the sensor data[cite: 48].
        """
        # P(X_{t}) = P(X_{t-1}) * T
        self.belief = self.belief @ self.T

    def update(self, board: Board, sensor_data: Tuple[Noise, int], worker_pos: Tuple[int, int]):
        """
        Updates the belief state based on sensor observations.
        
        Args:
            board: The game board containing the cell states.
            sensor_data: A tuple of (Noise Enum, Estimated Distance).
            worker_pos: The (x, y) location of the worker hearing the noise.
        """
        noise, est_dist = sensor_data
        
        # 1. Calculate actual Manhattan distance from worker to ALL 64 cells instantly
        wx, wy = worker_pos
        actual_dists = np.abs(self.grid_x_flat - wx) + np.abs(self.grid_y_flat - wy)
        
        # Fetch distance likelihoods: P(est_dist | actual_dists)
        # We cap actual_dists at 14 just in case, though math guarantees it on 8x8
        likelihood_dist = self.P_dist[np.clip(actual_dists, 0, 14), est_dist]
        
        # 2. Fetch noise likelihoods: P(noise | cell_type)
        # We extract the cell types for all 64 squares. 
        # Using the bitmasks directly is the fastest way to get the cell types.
        shifts = np.arange(64, dtype=np.uint64)
        primed_bits  = (np.uint64(board._primed_mask) >> shifts) & 1
        carpet_bits  = (np.uint64(board._carpet_mask) >> shifts) & 1
        blocked_bits = (np.uint64(board._blocked_mask) >> shifts) & 1
        
        # Construct an array of Cell Enums (0=Space, 1=Primed, 2=Carpet, 3=Blocked)
        cell_types = (primed_bits * Cell.PRIMED + carpet_bits * Cell.CARPET + blocked_bits * Cell.BLOCKED).astype(int)
        likelihood_noise = self.P_noise[cell_types, noise.value] #UPDATED
        
        # 3. Apply Bayes' Theorem [cite: 197, 198]
        # P(State | Obs) \propto P(State) * P(Obs | State)
        self.belief = self.belief * likelihood_dist * likelihood_noise
        
        # 4. Normalize the distribution to sum to 1.0 
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            # Fallback if probability vanishes (e.g. extreme sensor noise combined with edge cases)
            self.belief = np.ones(64, dtype=np.float32) / 64.0
            
    def get_belief_array(self) -> np.ndarray:
        """Returns the belief formatted as an 8x8 array for the Neural Network."""
        return self.belief.reshape((8, 8))