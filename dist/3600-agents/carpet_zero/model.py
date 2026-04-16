import torch
import torch.nn as nn
import torch.nn.functional as F
from game.enums import Direction
from game.move import Move

# (ResidualBlock remains exactly the same as before)
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)

class CarpetZeroNet(nn.Module):
    def __init__(self, spatial_channels: int = 9, scalar_features: int = 6, 
                 hidden_channels: int = 128, num_res_blocks: int = 8, 
                 board_size: int = 8, num_actions: int = 36):
        super().__init__()
        self.board_size = board_size
        self.total_input_channels = spatial_channels + scalar_features
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.total_input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
        ])
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, spatial_state: torch.Tensor, scalar_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spatial_state.size(0)
        scalar_broadcast = scalar_state.view(batch_size, -1, 1, 1).expand(
            batch_size, -1, self.board_size, self.board_size
        )
        x = torch.cat([spatial_state, scalar_broadcast], dim=1)
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

def index_to_move(index: int) -> Move:
    """Deterministically maps an integer [0, 35] to a valid board Move object."""
    if 0 <= index < 4:
        return Move.plain(Direction(index))
    elif 4 <= index < 8:
        return Move.prime(Direction(index - 4))
    elif 8 <= index < 36:
        carpet_idx = index - 8
        direction = Direction(carpet_idx // 7)
        roll_length = (carpet_idx % 7) + 1
        return Move.carpet(direction, roll=roll_length)
    else:
        raise ValueError(f"Action index {index} out of bounds.")