import numpy as np
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums
from .hmm import RatHMM
from .expectiminimax import get_best_move

class PlayerAgent:
    """
    The main tournament agent that integrates the Hidden Markov Model 
    and Expectiminimax search to play the game optimally.
    """

    def __init__(self, board: board.Board, transition_matrix: np.ndarray = None, time_left: Callable = None):
        """
        Initializes the agent and sets up the HMM for tracking the rat.
        """
        self.hmm = RatHMM(transition_matrix)
        self.is_first_turn = True

    def commentate(self) -> str:
        """
        Optional: Prints out commentary at the end of the game.
        """
        return "Search complete. Good game."

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> move.Move:
        """
        Executes the bot's turn logic:
        1. Synchronizes the HMM with the unobserved rat steps.
        2. Applies the current turn's sensor data to the HMM.
        3. Budgets time and calls the Expectiminimax search.
        """
        noise, estimated_distance = sensor_data
        my_worker = board.player_worker
        my_loc = my_worker.get_location()

        # --- 1. Synchronize the Hidden Markov Model ---

        # Handle rat respawn: reset belief to post-1000-step distribution
        if board.opponent_search[1] or board.player_search[1]:
            self.hmm.reset()

        # Between our last turn and now, the rat moved once (unobserved) before
        # the opponent's turn. Skip this on our very first turn as Player A
        # (turn_count == 0) since there was no prior opponent turn.
        if not self.is_first_turn:
            self.hmm.belief = np.dot(self.hmm.belief, self.hmm.T)

        self.is_first_turn = False

        # --- 2. Predict current turn and Update with Sensor Data ---
        self.hmm.predict_and_update(board, my_loc, noise, estimated_distance)

        # --- 3. Time Budgeting ---
        # Allocate roughly proportional time based on remaining turns, leaving a buffer
        remaining_time = time_left()
        turns_left = max(my_worker.turns_left, 1)
        
        # Aim to use an equal fraction of the remaining time, but cap it so we 
        # don't waste 20 seconds on a single move if we have a surplus.
        allocated_time = (remaining_time / turns_left) * 0.9
        allocated_time = min(allocated_time, 8.0) 
        
        # Hard safety margin to prevent losing via timeout
        allocated_time = min(allocated_time, remaining_time - 0.5)

        # --- 4. Search and Execute ---
        best_move = get_best_move(
            board=board,
            max_time=allocated_time,
            rat_belief=self.hmm.belief
        )

        return best_move