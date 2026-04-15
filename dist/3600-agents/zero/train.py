import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import pickle
import time
import os
import glob
from network import CarpetNet
from pipeline import HostReplayBuffer, train_step, generate_self_play_batch

CHECKPOINT_DIR = "checkpoints"

def save_checkpoint(gen: int, params, opt_state, loss: float):
    """Safely writes checkpoints atomically to prevent corruption during SLURM preemptions."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    temp_path = f"{CHECKPOINT_DIR}/temp_checkpoint.pkl"
    final_path = f"{CHECKPOINT_DIR}/checkpoint_gen_{gen}.pkl"
    
    chkpt = {'gen': gen, 'params': params, 'opt_state': opt_state, 'loss': loss}
    with open(temp_path, "wb") as f:
        pickle.dump(chkpt, f)
    os.replace(temp_path, final_path)

def load_latest_checkpoint(rng, dummy_input):
    """Resumes training from the most recent save state if available."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint_gen_*.pkl")
    
    if not checkpoints:
        params = CarpetNet().init(rng, dummy_input)['params']
        opt_state = optax.adamw(1e-3, 1e-4).init(params)
        return params, opt_state, 0
    else:
        latest = max(checkpoints, key=lambda x: int(x.split('_gen_')[1].split('.pkl')[0]))
        with open(latest, "rb") as f:
            chkpt = pickle.load(f)
        return chkpt['params'], chkpt['opt_state'], chkpt['gen'] + 1

def run_training():
    """Main execution loop linking the Replay Buffer and self-play mechanics."""
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 8, 8, 7)) 
    params, opt_state, start_gen = load_latest_checkpoint(rng, dummy_input)
    buffer = HostReplayBuffer(capacity=1_000_000)
    
    # Hyperparameters for the training loop
    GAMES_PER_GEN = 16       # Number of concurrent games to simulate per generation
    BATCH_SIZE = 32           # Size of the training minibatch
    TRAIN_STEPS_PER_GEN = 5   # How many gradient updates to perform per generation
    
    try:
        print(f"Starting AlphaZero Pipeline from Generation {start_gen}...")
        for gen in range(start_gen, 5000):
            start_time = time.time()
            rng, subkey = jax.random.split(rng)
            
            # 1. Generate Self-Play Data (Pure GPU)
            # This calls the massive JAX-compiled simulation loop
            states, policies, values = generate_self_play_batch(subkey, params, num_games=GAMES_PER_GEN)
            
            # 2. Push to CPU Replay Buffer
            buffer.add_batch(states, policies, values)
            
            # 3. Gradient Descent Loop (Pull from CPU -> Train on GPU)
            total_loss = 0.0
            if buffer.size > BATCH_SIZE * 5: # Wait until we have enough data to train
                for _ in range(TRAIN_STEPS_PER_GEN):
                    b_states, b_policies, b_values = buffer.sample_minibatch(BATCH_SIZE)
                    params, opt_state, loss = train_step(params, opt_state, b_states, b_policies, b_values)
                    total_loss += loss
                    
            avg_loss = total_loss / TRAIN_STEPS_PER_GEN if TRAIN_STEPS_PER_GEN > 0 else 0.0
            
            # 4. Save Checkpoints and Log
            if gen % 10 == 0:
                save_checkpoint(gen, params, opt_state, avg_loss)
                
            elapsed = time.time() - start_time
            print(f"Gen {gen} | Loss: {avg_loss:.4f} | Buffer: {buffer.size} | Time: {elapsed:.1f}s")
            
    finally:
        print("Saving emergency checkpoint...")
        save_checkpoint(99999, params, opt_state, 0.0)

if __name__ == "__main__":
    run_training()