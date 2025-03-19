import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc
import wandb

from Network import NeRFNetwork

wandb.init(project="NeRF", name="lego_train")

class NeRFDataset(Dataset):
    def __init__(self, ray_origins_path, ray_directions_path, rgb_path, points_3D_path=None):
        # Load ray origins and directions
        print("Loading ray origins...")
        self.ray_origins = np.load(ray_origins_path).copy()  # Adding .copy() to make writable
        print(f"Ray origins shape: {self.ray_origins.shape}")
        
        print("Loading ray directions...")
        self.ray_directions = np.load(ray_directions_path).copy()
        print(f"Ray directions shape: {self.ray_directions.shape}")
        
        print("Loading RGB values...")
        self.rgb_values = np.load(rgb_path).copy()
        print(f"RGB values shape: {self.rgb_values.shape}")
        
        # Extract images, height, width from the data shape
        # Assuming the original data shape is (num_images, H, W, 3)
        if len(self.ray_origins.shape) == 4:
            self.num_images, self.H, self.W = self.ray_origins.shape[:3]
        else:
            # If already flattened, you need to provide these values
            # Either hardcode them or pass them as parameters
            raise ValueError("Data is already flattened, but num_images, H, W not provided")
        
        print(f"Dataset has {self.num_images} images of size {self.H}x{self.W}")
        
        # If points_3D file is provided, load it
        if points_3D_path:
            print("Loading 3D points...")
            self.points_3D = np.load(points_3D_path)
            print(f"3D points shape: {self.points_3D.shape}")
            
            # Check if we need to flatten the points_3D array
            if len(self.points_3D.shape) == 5:  # Shape: (num_images, samples_per_ray, H, W, 3)
                print("Reshaping 3D points to match flattened rays...")
                # We need to reshape to (num_rays, samples_per_ray, 3)
                num_images, samples_per_ray, H, W, coords = self.points_3D.shape
                self.points_3D = self.points_3D.reshape(-1, samples_per_ray, coords)
                print(f"Reshaped 3D points shape: {self.points_3D.shape}")
            
            self.use_precomputed_points = True
        else:
            self.use_precomputed_points = False
            
        # Reshape to have each ray as a separate data point
        print("Reshaping data...")
        self.ray_origins = self.ray_origins.reshape(-1, 3)
        self.ray_directions = self.ray_directions.reshape(-1, 3)
        self.rgb_values = self.rgb_values.reshape(-1, 3)
        
        # Create index mapping from flattened to original indices
        print("Creating index mapping...")
        self.total_rays = self.ray_origins.shape[0]
        self.idx_mapping = np.zeros((self.total_rays, 3), dtype=np.int32)
        for img_idx in range(self.num_images):
            for h in range(self.H):
                for w in range(self.W):
                    flat_idx = img_idx * self.H * self.W + h * self.W + w
                    self.idx_mapping[flat_idx] = [img_idx, h, w]
        
        print(f"Dataset initialized with {self.total_rays} rays")
    
    def __len__(self):
        return self.total_rays
    
    def __getitem__(self, idx):
        ray_origin = self.ray_origins[idx]
        ray_direction = self.ray_directions[idx]
        rgb_value = self.rgb_values[idx]
        
        if self.use_precomputed_points:
            # Add a check to prevent index out of bounds
            if idx < len(self.points_3D):
                points_3D = self.points_3D[idx]
                return torch.from_numpy(ray_origin).float(), torch.from_numpy(ray_direction).float(), torch.from_numpy(rgb_value).float(), torch.from_numpy(points_3D).float()
            else:
                # Either disable use of precomputed points for this ray
                print(f"Warning: Index {idx} out of bounds for points_3D with size {len(self.points_3D)}")
                return torch.from_numpy(ray_origin).float(), torch.from_numpy(ray_direction).float(), torch.from_numpy(rgb_value).float()
        
        return torch.from_numpy(ray_origin).float(), torch.from_numpy(ray_direction).float(), torch.from_numpy(rgb_value).float()
    

# Define the function to compute accumulated transmittance
def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1) #Accumulated transmittance over all points (dim 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), #Transmission is 1 at first point of each ray
                    accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

# Define the render_from_presampled function
def render_from_presampled(nerf_model, ray_origins, ray_directions, points_3D, chunk_size=64):
    # Compute distances between consecutive points along each ray
    # Assuming points_3D shape is [batch_size, num_samples, 3]
    
    # Calculate distances between consecutive samples
    deltas = torch.norm(points_3D[:, 1:] - points_3D[:, :-1], dim=-1)

    # Ddeleting large memory chunks after usage
    points_diff = points_3D[:, 1:] - points_3D[:, :-1]
    deltas = torch.norm(points_diff, dim=-1)
    del points_diff

    # Add a large value for the last delta
    deltas = torch.cat([deltas, torch.ones(deltas.shape[0], 1, device=deltas.device) * 1e10], dim=-1)

    # Process in chunks to save memory
    num_rays = points_3D.shape[0]
    num_samples = points_3D.shape[1]
    
    colors_list = []
    sigma_list = []
    
    for i in range(0, num_rays, chunk_size):
        chunk_points = points_3D[i:i+chunk_size]
        chunk_dirs = ray_directions[i:i+chunk_size].unsqueeze(1).expand(-1, num_samples, -1)
        
        # Flatten for network processing
        chunk_points_flat = chunk_points.reshape(-1, 3)
        chunk_dirs_flat = chunk_dirs.reshape(-1, 3)
        
        # Process through network
        chunk_colors, chunk_sigma = nerf_model(chunk_points_flat, chunk_dirs_flat)
        
        # Reshape back
        chunk_colors = chunk_colors.reshape(-1, num_samples, 3)
        chunk_sigma = chunk_sigma.reshape(-1, num_samples)
        
        colors_list.append(chunk_colors)
        sigma_list.append(chunk_sigma)
    
    # Combine results
    colors = torch.cat(colors_list, dim=0)
    sigma = torch.cat(sigma_list, dim=0)
    
    # Calculate alpha from sigma and deltas
    alpha = 1 - torch.exp(-sigma * deltas)
    
    # Compute weights using accumulated transmittance
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    
    # Compute final pixel colors
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(dim=1).sum(dim=-1)
    
    # Return final color (adding white background)
    return c + 1 - weight_sum.unsqueeze(-1)
    #return c 

# Define the training function
def train(nerf_model, optimizer, scheduler, data_loader, device='cuda', nb_epochs=int(1e5)):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_loss = 0
        for i,batch in enumerate(data_loader):
            print(f"Processing Batch {i}")
            #if len(batch) == 2:  # If using precomputed points
            print("Using Precomputed data")
            ray_origins, ray_directions, ground_truth_px_values, points_3D = batch
            ray_origins = ray_origins.to(device)
            ray_directions = ray_directions.to(device)
            ground_truth_px_values = ground_truth_px_values.to(device)
            #points_3D = points_3D.to(device)

            print("Vol Rendering start")
            # regenerated_px_values = render_from_presampled(
            #     nerf_model, ray_origins, ray_directions, points_3D
            # )
            regenerated_px_values = render_rays(nerf_model,ray_origins,ray_directions,hn=2,hf=6,nb_bins=192)
            print("Volume Rendering Fin")
            
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #wandb.log({"epoch_unsup": Epochs, "loss_unsup": avg_epoch_loss, "acc_unsup": avg_epoch_accuracy, "epe_unsup": avg_epe_loss})
        
        training_loss.append(epoch_loss / len(data_loader))
        scheduler.step()
        
        # Print progress
        #if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {training_loss[-1]:.4f}")
        wandb.log({"Epochs":epoch+1, "Loss":training_loss[-1]})
            
        # Save model checkpoint
        if (epoch + 1) % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': nerf_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_loss[-1],
            }, f"checkpoints/nerf_model_epoch_{epoch+1}.pth")

            
    return training_loss

def gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a NeRF model')
    parser.add_argument('--data_dir', type=str, default='/home/sviswasam/cv/Ship_Data_Processed/', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=16, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the NeRF network')
    parser.add_argument('--use_precomputed', action='store_true', help='Use precomputed 3D points')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training (cuda/cpu)')
    args = parser.parse_args()
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    print("Arguments Parsed")

    # Check gpu usage after this
    gpu_usage()
    
    # Define paths to data
    ray_origins_path = os.path.join(args.data_dir, 'ray_origins.npy')
    ray_directions_path = os.path.join(args.data_dir, 'ray_directions.npy')
    rgb_path = os.path.join(args.data_dir, 'rgb_values.npy')
    points_3D_path = os.path.join(args.data_dir, 'points_3D.npy') 

    gpu_usage()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = NeRFDataset(ray_origins_path, ray_directions_path, rgb_path, points_3D_path)
    #dataset = NeRFDataset(ray_origins_path, ray_directions_path, rgb_path, points_3D_path, 
                     #num_images=100, H=800, W=800) 
    
    print("Started Dataloading...")
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    print("Data loaded Successfully")
    
    # Create model
    nerf_model = NeRFNetwork(
        #hidden_dim=args.hidden_dim,
        #use_viewdirs=True  # Assuming the model uses view directions
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    
    # Train model
    print("Starting training...")
    training_loss = train(
        nerf_model, 
        optimizer, 
        scheduler, 
        data_loader, 
        device=device, 
        nb_epochs=args.epochs
    )

    wandb.finish()

    print("Training finished. Saving model...")
    
    # Save final model
    torch.save({
        'model_state_dict': nerf_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': training_loss[-1],
    }, "checkpoints/nerf_model_ship.pth")
    
    # # Plot training loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(training_loss)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('training_loss.png')
    # plt.close()
    
    print("Training complete! Final model saved.")
