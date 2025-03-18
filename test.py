import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from train import render_from_presampled,compute_accumulated_transmittance
from Network import NeRFNetwork

def test_from_json(model, json_path, device='cpu', chunk_size=1024, samples_per_ray=64, near=2.0, far=6.0, output_dir='novel_views'):
    """
    Generate novel views from camera poses in a JSON file using presampled points
    
    Args:
        model: Trained NeRF model
        json_path: Path to JSON file with camera parameters
        device: Device to run on
        chunk_size: Number of rays to process at once
        samples_per_ray: Number of sample points along each ray
        near, far: Near and far plane distances
        output_dir: Directory to save output images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera parameters from JSON
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    
    # Process each camera in the JSON
    for i, camera in enumerate(camera_data['frames']):
        # Extract camera pose (transform_matrix in NeRF datasets)
        pose = np.array(camera['transform_matrix'])
        
        # Extract camera parameters (adjust based on your JSON structure)
        if 'focal_length' in camera:
            focal_length = camera['focal_length']
        else:
            focal_length = 1000.0  # Default value
            
        if 'width' in camera and 'height' in camera:
            W, H = camera['width'], camera['height']
        else:
            W, H = 400, 400  # Default resolution
        
        print(f"Rendering view {i+1}/{len(camera_data['frames'])} with resolution {W}x{H}")
        
        # Generate rays for this camera
        ray_origins = []
        ray_directions = []
        
        # Compute the ray origins and directions
        for j in range(H):
            for k in range(W):
                # Convert pixel coordinates to NDC space
                x = (2 * (k + 0.5) / W - 1) * W / focal_length
                y = (1 - 2 * (j + 0.5) / H) * H / focal_length
                
                # Create ray direction in camera space
                direction = np.array([x, y, -1.0])
                
                # Transform to world space
                direction_world = np.dot(pose[:3, :3], direction)
                direction_world = direction_world / np.linalg.norm(direction_world)
                
                # Ray origin is the camera position
                origin_world = pose[:3, 3]
                
                ray_origins.append(origin_world)
                ray_directions.append(direction_world)
        
        # Convert to tensors
        ray_origins = torch.tensor(ray_origins, dtype=torch.float32).to(device)
        ray_directions = torch.tensor(ray_directions, dtype=torch.float32).to(device)
        
        # Total number of rays
        total_rays = ray_origins.shape[0]
        
        # Initialize image storage
        rendered_pixels = []
        
        # Process in batches
        for ray_idx in range(0, total_rays, chunk_size):
            # Get current batch of rays
            batch_origins = ray_origins[ray_idx:ray_idx+chunk_size]
            batch_directions = ray_directions[ray_idx:ray_idx+chunk_size]
            
            # Generate sample points along each ray
            # Create sample points along the ray from near to far
            t_vals = torch.linspace(0, 1, samples_per_ray).to(device)
            z_vals = near * (1 - t_vals) + far * t_vals
            
            # Get sample points along each ray
            # [batch_size, samples_per_ray, 3]
            points_3D = batch_origins.unsqueeze(1) + batch_directions.unsqueeze(1) * z_vals.unsqueeze(0).unsqueeze(-1)
            
            # Render this batch with presampled points
            with torch.no_grad():
                rendered_batch = render_from_presampled(
                    model, 
                    batch_origins, 
                    batch_directions, 
                    points_3D, 
                    chunk_size=min(64, len(batch_origins))  # Use smaller chunk size for rendering
                )
            
            rendered_pixels.append(rendered_batch.cpu())
            
            # Print progress
            if ray_idx % (10 * chunk_size) == 0:
                print(f"  Processed {ray_idx}/{total_rays} rays")
        
        # Combine all pixels and reshape to image
        rendered_image = torch.cat(rendered_pixels, dim=0).reshape(H, W, 3).numpy()
        
        # Save the image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(np.clip(rendered_image, 0, 1))
        # plt.axis('off')
        
        # Get filename from JSON if available
        if 'file_path' in camera:
            filename = os.path.basename(camera['file_path']).split('.')[0]
        else:
            filename = f"view_{i:03d}"
            
        # plt.savefig(f'{output_dir}/{filename}.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        
        # Also save raw numpy array
        np.save(f'{output_dir}/{filename}.npy', rendered_image)
        
        print(f"Saved novel view as {output_dir}/{filename}.png")

def test_specific_frame(model, json_path, target_filename="r_30.png", device='cpu', chunk_size=1024, 
                        samples_per_ray=64, near=2.0, far=6.0, output_dir='novel_views'):
    """
    Generate a novel view for a specific frame (e.g., r_30.png) from camera poses in a JSON file
    
    Args:
        model: Trained NeRF model
        json_path: Path to JSON file with camera parameters
        target_filename: The specific file to render (e.g., "r_30.png")
        device: Device to run on
        chunk_size: Number of rays to process at once
        samples_per_ray: Number of sample points along each ray
        near, far: Near and far plane distances
        output_dir: Directory to save output images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera parameters from JSON
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    
    # Find the specific camera for target_filename
    target_camera = None
    for camera in camera_data['frames']:
        file_path = camera.get('file_path', '')
        if target_filename in file_path:
            target_camera = camera
            break
    
    if target_camera is None:
        print(f"Could not find frame with filename {target_filename} in the JSON file")
        return
    
    # Extract camera pose
    pose = np.array(target_camera['transform_matrix'])
    
    # Extract camera parameters (adjust based on your JSON structure)
    if 'focal_length' in target_camera:
        focal_length = target_camera['focal_length']
    else:
        focal_length = 1000.0  # Default value
        
    if 'width' in target_camera and 'height' in target_camera:
        W, H = target_camera['width'], target_camera['height']
    else:
        W, H = 400, 400  # Default resolution
    
    print(f"Rendering view {target_filename} with resolution {W}x{H}")
    
    # Generate rays for this camera
    ray_origins = []
    ray_directions = []
    
    # Compute the ray origins and directions
    for j in range(H):
        for k in range(W):
            # Convert pixel coordinates to NDC space
            x = (2 * (k + 0.5) / W - 1) * W / focal_length
            y = (1 - 2 * (j + 0.5) / H) * H / focal_length
            
            # Create ray direction in camera space
            direction = np.array([x, y, -1.0])
            
            # Transform to world space
            direction_world = np.dot(pose[:3, :3], direction)
            direction_world = direction_world / np.linalg.norm(direction_world)
            
            # Ray origin is the camera position
            origin_world = pose[:3, 3]
            
            ray_origins.append(origin_world)
            ray_directions.append(direction_world)
    
    # Convert to tensors
    ray_origins = torch.tensor(ray_origins, dtype=torch.float32).to(device)
    ray_directions = torch.tensor(ray_directions, dtype=torch.float32).to(device)
    
    # Total number of rays
    total_rays = ray_origins.shape[0]
    
    # Initialize image storage
    rendered_pixels = []
    
    # Process in batches
    for ray_idx in range(0, total_rays, chunk_size):
        # Get current batch of rays
        batch_origins = ray_origins[ray_idx:ray_idx+chunk_size]
        batch_directions = ray_directions[ray_idx:ray_idx+chunk_size]
        
        # Generate sample points along each ray
        t_vals = torch.linspace(0, 1, samples_per_ray).to(device)
        z_vals = near * (1 - t_vals) + far * t_vals
        
        # Get sample points along each ray
        points_3D = batch_origins.unsqueeze(1) + batch_directions.unsqueeze(1) * z_vals.unsqueeze(0).unsqueeze(-1)
        
        # Render this batch with presampled points
        with torch.no_grad():
            rendered_batch = render_from_presampled(
                model, 
                batch_origins, 
                batch_directions, 
                points_3D, 
                chunk_size=min(64, len(batch_origins))
            )
        
        rendered_pixels.append(rendered_batch.cpu())
        
        # Print progress
        if ray_idx % (10 * chunk_size) == 0:
            print(f"  Processed {ray_idx}/{total_rays} rays")
    
    # Combine all pixels and reshape to image
    rendered_image = torch.cat(rendered_pixels, dim=0).reshape(H, W, 3).numpy()
    
    # Save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(rendered_image, 0, 1))
    plt.axis('off')
    
    # Generate output filename
    output_filename = 'r_30'
    plt.savefig(f'{output_dir}/{output_filename}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Also save raw numpy array
    np.save(f'{output_dir}/{output_filename}.npy', rendered_image)
    
    print(f"Saved novel view as {output_dir}/{output_filename}.png")

def load_model(model_path, device='cpu'):
    # Initialize your NeRF model architecture
    model = NeRFNetwork()  # Replace with your actual model initialization
    
    # Load the saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Load your trained model
    model = load_model("checkpoints/nerf_model_final.pth")

    # Test with your JSON file
    test_from_json(model, "Data/lego/transforms_test.json", near=2.0, far=6.0, samples_per_ray=64)
