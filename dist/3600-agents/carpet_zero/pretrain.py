import sys
import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Pathing fix (same as before)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
engine_path = os.path.join(root_dir, "engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

# 2. Local Imports
try:
    from .model import CarpetZeroNet
    from .self_play import AlphaZeroDataset
    from .json_parser import parse_match_json
except ImportError:
    from model import CarpetZeroNet
    from self_play import AlphaZeroDataset
    from json_parser import parse_match_json

def train_on_expert_data(matches_dir: str, epochs: int = 20, batch_size: int = 128):
    """
    Loads all JSON matches in the directory, parses them into AlphaZero tensors, 
    and trains the Neural Network to mimic the expert moves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Find all match JSONs
    json_files = glob.glob(os.path.join(matches_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {matches_dir}!")
        return

    print(f"Found {len(json_files)} match files. Parsing data...")

    # 2. Parse all files into a giant dataset
    all_training_data = []
    for j_file in json_files:
        try:
            match_data = parse_match_json(j_file)
            all_training_data.extend(match_data)
        except Exception as e:
            print(f"Error parsing {j_file}: {e}")

    print(f"Successfully generated {len(all_training_data)} training samples!")

    # 3. Setup PyTorch DataLoaders and Model
    dataset = AlphaZeroDataset(all_training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CarpetZeroNet().to(device)
    model_path = "best_model.pth"
    
    # Load existing weights if they exist
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing best_model.pth")

    # Use a slightly lower learning rate for supervised fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        
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
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/num_batches:.4f} "
              f"(Policy: {policy_loss_sum/num_batches:.4f}, Value: {value_loss_sum/num_batches:.4f})")
        
    torch.save(model.state_dict(), model_path)
    print("Updated and saved best_model.pth!")

if __name__ == "__main__":
    # Point this to your matches folder
    matches_folder = os.path.abspath(os.path.join(root_dir, "3600-agents/matches"))
    train_on_expert_data(matches_folder, epochs=10)