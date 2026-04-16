import numpy as np
import torch
from game.board import Board
from game.enums import ALLOWED_TIME, MAX_TURNS_PER_PLAYER

class StateSerializer:
    def __init__(self):
        self._shifts = np.arange(64, dtype=np.uint64)
        
    def _bitboard_to_array(self, mask: int) -> np.ndarray:
        bits = (np.uint64(mask) >> self._shifts) & 1
        return bits.astype(np.float32).reshape((8, 8))

    def _coord_to_array(self, coord: tuple[int, int] | None) -> np.ndarray:
        arr = np.zeros((8, 8), dtype=np.float32)
        if coord is not None and coord[0] >= 0 and coord[1] >= 0:
            arr[coord[1], coord[0]] = 1.0
        return arr

    def serialize_single(self, board: Board, hmm_belief: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Now only 8 spatial channels
        spatial_channels = np.stack([
            self._coord_to_array(board.player_worker.get_location()),
            self._coord_to_array(board.opponent_worker.get_location()),
            self._bitboard_to_array(board._blocked_mask),
            self._bitboard_to_array(board._space_mask),
            self._bitboard_to_array(board._primed_mask),
            self._bitboard_to_array(board._carpet_mask),
            self._coord_to_array(board.player_search[0]),
            self._coord_to_array(board.opponent_search[0]),
            hmm_belief.astype(np.float32) # <-- Added 9th channel
        ])
        
        scalar_features = np.array([
            board.player_worker.get_points() / 50.0,
            board.opponent_worker.get_points() / 50.0,
            board.player_worker.time_left / float(ALLOWED_TIME),
            board.opponent_worker.time_left / float(ALLOWED_TIME),
            board.player_worker.turns_left / float(MAX_TURNS_PER_PLAYER),
            board.opponent_worker.turns_left / float(MAX_TURNS_PER_PLAYER)
        ], dtype=np.float32)
        
        spatial_tensor = torch.from_numpy(spatial_channels).unsqueeze(0)
        scalar_tensor = torch.from_numpy(scalar_features).unsqueeze(0)
        
        return spatial_tensor, scalar_tensor