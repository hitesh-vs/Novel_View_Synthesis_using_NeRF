# import numpy as np
# import torch
# import torch.nn as nn
# import tqdm
# from Network import NeRFNetwork

# def compute_accumulated_transmittance(alphas):
#     accumulated_transmittance = torch.cumprod(alphas, 1) #Accumulated transmittance over all points (dim 1)
#     return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), #Transmission is 1 at first point of each ray
#                     accumulated_transmittance[:, :-1]), dim=-1)

# def render_from_presampled(nerf_model, ray_origins, ray_directions, points_3D):
#     # Compute distances between consecutive points along each ray
#     # Assuming points_3D shape is [batch_size, num_samples, 3]
    
#     # Calculate distances between consecutive samples
#     deltas = torch.norm(points_3D[:, 1:] - points_3D[:, :-1], dim=-1)
#     # Add a large value for the last delta
#     deltas = torch.cat([deltas, torch.ones(deltas.shape[0], 1, device=deltas.device) * 1e10], dim=-1)
    
#     # Reshape ray_directions to match the points
#     num_samples = points_3D.shape[1]
#     ray_directions_expanded = ray_directions.unsqueeze(1).expand(-1, num_samples, -1)
    
#     # Predict colors and densities
#     points_flat = points_3D.reshape(-1, 3)
#     directions_flat = ray_directions_expanded.reshape(-1, 3)
#     colors, sigma = nerf_model(points_flat, directions_flat)
    
#     # Reshape back
#     colors = colors.reshape(points_3D.shape[0], num_samples, -1)  # [batch_size, num_samples, 3]
#     sigma = sigma.reshape(points_3D.shape[0], num_samples)        # [batch_size, num_samples]
    
#     # Calculate alpha from sigma and deltas
#     alpha = 1 - torch.exp(-sigma * deltas)
    
#     # Compute weights using accumulated transmittance
#     weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    
#     # Compute final pixel colors
#     c = (weights * colors).sum(dim=1)
#     weight_sum = weights.sum(dim=1).sum(dim=-1)
    
#     # Return final color (adding white background)
#     return c + 1 - weight_sum.unsqueeze(-1)

# class NeRFDataset(Dataset):
#     def __init__(self, ray_origins_path, ray_directions_path, rgb_path, points_3D_path=None):
#         # Load ray origins and directions
#         self.ray_origins = np.load(ray_origins_path)  # [num_images, H, W, 3]
#         self.ray_directions = np.load(ray_directions_path)  # [num_images, H, W, 3]
#         self.rgb_values = np.load(rgb_path)  # [num_images, H, W, 3]
        
#         # Get shapes
#         self.num_images, self.H, self.W, _ = self.ray_origins.shape
        
#         # If points_3D file is provided, load it
#         if points_3D_path:
#             # Use memory mapping for large files
#             self.points_3D = np.load(points_3D_path, mmap_mode='r')  # [num_images, num_samples, H, W, 3]
#             self.num_samples = self.points_3D.shape[1]
#             self.use_precomputed_points = True
#         else:
#             self.use_precomputed_points = False
            
#         # Reshape to have each ray as a separate data point
#         self.ray_origins = self.ray_origins.reshape(-1, 3)
#         self.ray_directions = self.ray_directions.reshape(-1, 3)
#         self.rgb_values = self.rgb_values.reshape(-1, 3)
        
#         # Create index mapping from flattened to original indices
#         self.total_rays = self.ray_origins.shape[0]
#         self.idx_mapping = np.zeros((self.total_rays, 3), dtype=np.int32)
#         for img_idx in range(self.num_images):
#             for h in range(self.H):
#                 for w in range(self.W):
#                     flat_idx = img_idx * self.H * self.W + h * self.W + w
#                     self.idx_mapping[flat_idx] = [img_idx, h, w]
        
#     def __len__(self):
#         return self.total_rays
    
#     def __getitem__(self, idx):
#         # Get ray origin, direction and RGB
#         ray_origin = self.ray_origins[idx]
#         ray_direction = self.ray_directions[idx]
#         rgb = self.rgb_values[idx]
        
#         # Combine into a single sample
#         sample = np.concatenate([ray_origin, ray_direction, rgb])
        
#         if self.use_precomputed_points:
#             # Get the original indices
#             img_idx, h, w = self.idx_mapping[idx]
#             # Extract the points for this ray
#             points = self.points_3D[img_idx, :, h, w, :]
#             return sample, points
#         else:
#             return sample

# def train(nerf_model, optimizer, scheduler, data_loader, device='cuda', nb_epochs=int(1e5)):
#     training_loss = []
#     for epoch in tqdm(range(nb_epochs)):
#         for batch in data_loader:
#             if len(batch) == 2:  # If using precomputed points
#                 data, points_3D = batch
#                 ray_origins = data[:, :3].to(device)
#                 ray_directions = data[:, 3:6].to(device)
#                 ground_truth_px_values = data[:, 6:].to(device)
#                 points_3D = points_3D.to(device)
                
#                 regenerated_px_values = render_from_presampled(
#                     nerf_model, ray_origins, ray_directions, points_3D
#                 )
            
#             loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             training_loss.append(loss.item())
#         scheduler.step()
#     return training_loss

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc

from Network import NeRFNetwork

class NeRFDataset(Dataset):
    # def __init__(self, ray_origins_path, ray_directions_path, rgb_path, points_3D_path=None):
    #     # Load ray origins and directions
    #     self.ray_origins = np.load(ray_origins_path).copy()  # Adding .copy() to make writable
    #     self.ray_directions = np.load(ray_directions_path).copy()
    #     self.rgb_values = np.load(rgb_path).copy()
        
    #     # If points_3D file is provided, load it
    #     if points_3D_path:
    #         # Modified: don't use mmap_mode
    #         self.points_3D = np.load(points_3D_path)
    #         # If memory is an issue, you can load small chunks at a time instead
    #         self.num_samples = self.points_3D.shape[1]
    #         self.use_precomputed_points = True
    #     else:
    #         self.use_precomputed_points = False
            
    #     # Reshape to have each ray as a separate data point
    #     self.ray_origins = self.ray_origins.reshape(-1, 3)
    #     self.ray_directions = self.ray_directions.reshape(-1, 3)
    #     self.rgb_values = self.rgb_values.reshape(-1, 3)
        
    #     # Create index mapping from flattened to original indices
    #     self.total_rays = self.ray_origins.shape[0]
    #     self.idx_mapping = np.zeros((self.total_rays, 3), dtype=np.int32)
    #     for img_idx in range(self.num_images):
    #         for h in range(self.H):
    #             for w in range(self.W):
    #                 flat_idx = img_idx * self.H * self.W + h * self.W + w
    #                 self.idx_mapping[flat_idx] = [img_idx, h, w]

    def __init__(self, ray_origins_path, ray_directions_path, rgb_path, points_3D_path=None):
        # Load ray origins and directions
        self.ray_origins = np.load(ray_origins_path).copy()  # Adding .copy() to make writable
        self.ray_directions = np.load(ray_directions_path).copy()
        self.rgb_values = np.load(rgb_path).copy()
        
        # If points_3D file is provided, load it
        if points_3D_path:
            # Modified: don't use mmap_mode
            self.points_3D = np.load(points_3D_path)
            # If memory is an issue, you can load small chunks at a time instead
            self.num_samples = self.points_3D.shape[1]
            self.use_precomputed_points = True
        else:
            self.use_precomputed_points = False
        
        # For single image processing
        self.num_images = 1
        
        # Reshape to have each ray as a separate data point
        self.ray_origins = self.ray_origins.reshape(-1, 3)
        self.ray_directions = self.ray_directions.reshape(-1, 3)
        self.rgb_values = self.rgb_values.reshape(-1, 3)
        
        # For single image, H and W should be derived from data dimensions
        self.H = int(np.sqrt(self.ray_origins.shape[0]))
        self.W = self.H  # Assuming square image, otherwise adjust accordingly
        
        # Create index mapping from flattened to original indices
        self.total_rays = self.ray_origins.shape[0]
        self.idx_mapping = np.zeros((self.total_rays, 3), dtype=np.int32)
        
        # Only one image (img_idx = 0)
        for h in range(self.H):
            for w in range(self.W):
                flat_idx = h * self.W + w
                self.idx_mapping[flat_idx] = [0, h, w]
        
    def __len__(self):
        return self.total_rays
    
    def __getitem__(self, idx):
        # Get ray origin, direction and RGB
        ray_origin = self.ray_origins[idx]
        ray_direction = self.ray_directions[idx]
        rgb = self.rgb_values[idx]
        
        # Combine into a single sample
        sample = np.concatenate([ray_origin, ray_direction, rgb])
        
        if self.use_precomputed_points:
            # Get the original indices
            img_idx, h, w = self.idx_mapping[idx]
            # Extract the points for this ray - make a copy to ensure writability
            points = np.array(self.points_3D[img_idx, :, h, w, :], copy=True)
            return sample, points
        else:
            return sample

# Define the function to compute accumulated transmittance
def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1) #Accumulated transmittance over all points (dim 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), #Transmission is 1 at first point of each ray
                    accumulated_transmittance[:, :-1]), dim=-1)

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
    
    # Rest of your rendering code...
    
    # Calculate alpha from sigma and deltas
    alpha = 1 - torch.exp(-sigma * deltas)
    
    # Compute weights using accumulated transmittance
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    
    # Compute final pixel colors
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(dim=1).sum(dim=-1)
    
    # Return final color (adding white background)
    return c + 1 - weight_sum.unsqueeze(-1)

# Define the training function
def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', nb_epochs=int(1e5)):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_loss = 0
        for i,batch in enumerate(data_loader):
            print(f"Processing Batch {i}")
            if len(batch) == 2:  # If using precomputed points
                print("Using Precomputed data")
                data, points_3D = batch
                ray_origins = data[:, :3].to(device)
                ray_directions = data[:, 3:6].to(device)
                ground_truth_px_values = data[:, 6:].to(device)
                points_3D = points_3D.to(device)
                
                print("Vol Rendering start")
                regenerated_px_values = render_from_presampled(
                    nerf_model, ray_origins, ray_directions, points_3D
                )
                print("Volume Rendering Fin")
            
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        training_loss.append(epoch_loss / len(data_loader))
        scheduler.step()
        
        # Print progress
        #if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {training_loss[-1]:.4f}")
            
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
    parser.add_argument('--data_dir', type=str, default='/home/sviswasam/cv/Lego_Data_Processed/', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30000, help='Number of epochs to train')
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
    
    print("Started Dataloading...")
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("Data loaded Successfully")
    
    # Create model
    nerf_model = NeRFNetwork(
        #hidden_dim=args.hidden_dim,
        #use_viewdirs=True  # Assuming the model uses view directions
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
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

    print("Training finished. Saving model...")
    
    # Save final model
    torch.save({
        'model_state_dict': nerf_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': training_loss[-1],
    }, "checkpoints/nerf_model_final.pth")
    
    # # Plot training loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(training_loss)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('training_loss.png')
    # plt.close()
    
    print("Training complete! Final model saved.")
