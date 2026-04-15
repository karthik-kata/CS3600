import jax.numpy as jnp
import flax.linen as nn

class ResBlock(nn.Module):
    """A standard residual block for deep feature extraction without vanishing gradients."""
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        return nn.relu(x + residual)

class CarpetNet(nn.Module):
    """A highly compressed ResNet optimized for the strict 4-minute time limit."""
    
    @nn.compact
    def __call__(self, x):
        # Initial spatial feature mapping
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Micro-ResNet backbone (3 blocks for speed)
        for _ in range(3):
            x = ResBlock(features=64)(x)
            
        # Policy Head: Predicts the strength of the 100 possible actions
        policy = nn.Conv(features=2, kernel_size=(1, 1))(x)
        policy = policy.reshape((policy.shape[0], -1)) 
        policy = nn.Dense(features=100)(policy) # Maps to Plain, Prime, Carpet, Search
        
        # Value Head: Predicts win/loss from current state (-1.0 to 1.0)
        value = nn.Conv(features=1, kernel_size=(1, 1))(x)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(features=64)(value)
        value = nn.relu(value)
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value) 
        
        return policy, value