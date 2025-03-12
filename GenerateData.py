import json
import numpy as np
import os
import torch
from PIL import Image

# Define the correct path to the JSON file
current_dir = os.path.dirname(os.getcwd())  # Get current directory
relative_path = os.path.join("nerf", "Data", "lego", "transforms_train.json")
full_path = os.path.join(current_dir, relative_path)

# Create output directory
output_dir = os.path.join(current_dir, "Lego_Data_Processed")
os.makedirs(output_dir, exist_ok=True)

# Load JSON data
with open(full_path, "r") as file:
    data = json.load(file)

# Extract camera intrinsics (FOV to focal length)
fov_x = data["camera_angle_x"]
H, W = 800, 800  # Image resolution
focal_length = W / (2 * np.tan(fov_x / 2))  # Compute focal length from FOV

# Number of sample points per ray
num_samples = 64
depth_samples = np.linspace(0.1, 5.0, num=num_samples)  # Sample depths

# Total number of images to process
num_images = len(data["frames"])

# Option 1: Process and save each image in full resolution with CUDA - Single consolidated output
def process_individual_images():
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move depth samples to GPU
    depth_samples_tensor = torch.tensor(depth_samples, device=device).float()
    
    # Prepare arrays to hold all data
    ray_origins_array = np.zeros((num_images, H, W, 3), dtype=np.float32)
    ray_directions_array = np.zeros((num_images, H, W, 3), dtype=np.float32)
    rgb_values_array = np.zeros((num_images, H, W, 3), dtype=np.float32)  # New array for RGB values
    
    # We'll build the points_3D array incrementally to save memory
    points_3D_shape = (num_images, num_samples, H, W, 3)
    points_3D_total_size = np.prod(points_3D_shape)
    print(f"Total size of points_3D array: {points_3D_total_size * 4 / (1024**3):.2f} GB")
    
    # Initialize with memmap to avoid memory issues
    points_3D_file = os.path.join(output_dir, "points_3D.npy")
    points_3D_memmap = np.lib.format.open_memmap(
        points_3D_file, 
        mode='w+', 
        dtype=np.float32, 
        shape=points_3D_shape
    )
    
    for i, frame in enumerate(data["frames"][:num_images]):
        print(f"Processing Image {i+1}/{num_images} (full resolution with CUDA)")

        # Extract transformation matrix and convert to tensor
        transformation_matrix = torch.tensor(
            frame["transform_matrix"], device=device
        ).float()
        
        camera_position = transformation_matrix[:3, 3]
        rotation_matrix = transformation_matrix[:3, :3]

        # Create a meshgrid of pixel coordinates
        k, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing='xy'
        )

        # Normalize pixel coordinates
        k = (k - W / 2) / focal_length
        j = -(j - H / 2) / focal_length

        # Define ray directions in camera space
        ones = torch.ones_like(k)
        ray_directions_camera = torch.stack([k, j, -ones], dim=-1)

        # Convert ray directions to world space
        ray_directions_world = torch.matmul(ray_directions_camera, rotation_matrix.T)
        ray_directions_world /= torch.norm(ray_directions_world, dim=-1, keepdim=True)

        # Broadcast camera position
        ray_origins = camera_position.expand_as(ray_directions_world)

        # Load the RGB image
        image_path = os.path.join(current_dir, "nerf", "Data", "lego", frame["file_path"] + ".png")
        image = np.array(Image.open(image_path))

        # Check if image has an alpha channel
        if image.shape[-1] == 4:
            # Convert RGBA to RGB by removing the alpha channel
            image = image[:,:,:3]

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Store RGB values
        rgb_values_array[i] = image

        # Move results to CPU and store in the consolidated arrays
        ray_origins_array[i] = ray_origins.cpu().numpy()
        ray_directions_array[i] = ray_directions_world.cpu().numpy()
        
        # Process 3D points in chunks on GPU to save memory
        chunk_size = 50  # Process 50 rows at a time to avoid OOM
        for row_start in range(0, H, chunk_size):
            row_end = min(row_start + chunk_size, H)
            
            # Get chunk of rays
            chunk_origins = ray_origins[row_start:row_end]
            chunk_directions = ray_directions_world[row_start:row_end]
            
            # Reshape for broadcasting
            origins_expanded = chunk_origins.unsqueeze(0)  # [1, chunk_size, W, 3]
            directions_expanded = chunk_directions.unsqueeze(0)  # [1, chunk_size, W, 3]
            depths_expanded = depth_samples_tensor.view(-1, 1, 1, 1)  # [D, 1, 1, 1]
            
            # Calculate points - this happens entirely on GPU
            chunk_points_3D = origins_expanded + depths_expanded * directions_expanded
            
            # Move to CPU and save to memmap
            chunk_points_3D_np = chunk_points_3D.cpu().numpy()
            points_3D_memmap[i, :, row_start:row_end, :, :] = chunk_points_3D_np
            
            # Free GPU memory
            del chunk_points_3D, origins_expanded, directions_expanded
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save the consolidated arrays for ray origins and directions
    print("\nSaving consolidated data arrays...")
    np.save(os.path.join(output_dir, "ray_origins.npy"), ray_origins_array)
    np.save(os.path.join(output_dir, "ray_directions.npy"), ray_directions_array)
    np.save(os.path.join(output_dir, "rgb_values.npy"), rgb_values_array)  # Save RGB values
    # points_3D is already saved as a memmap
    
    # Create a flattened dataset for training
    print("\nCreating training dataset...")
    ray_origins_flat = ray_origins_array.reshape(-1, 3)
    ray_directions_flat = ray_directions_array.reshape(-1, 3)
    rgb_values_flat = rgb_values_array.reshape(-1, 3)
    
    # Combine into a single array for training
    training_data = np.concatenate([
        ray_origins_flat, 
        ray_directions_flat,
        rgb_values_flat
    ], axis=1)
    
    # Save training data
    np.save(os.path.join(output_dir, "training_data.npy"), training_data)
    
    print(f"Ray Origins: {ray_origins_array.shape} → saved as 'ray_origins.npy'")
    print(f"Ray Directions: {ray_directions_array.shape} → saved as 'ray_directions.npy'")
    print(f"RGB Values: {rgb_values_array.shape} → saved as 'rgb_values.npy'")
    print(f"3D Sample Points: {points_3D_shape} → saved as 'points_3D.npy'")
    print(f"Training Data: {training_data.shape} → saved as 'training_data.npy'")

# Option 2: Process with downsampling and CUDA acceleration - Single consolidated output
def process_with_downsampling(downsample_factor=4):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reduced resolution
    H_down = H // downsample_factor
    W_down = W // downsample_factor
    
    # Move depth samples to GPU
    depth_samples_tensor = torch.tensor(depth_samples, device=device).float()
    
    # Prepare arrays to hold all data
    ray_origins_array = np.zeros((num_images, H_down, W_down, 3), dtype=np.float32)
    ray_directions_array = np.zeros((num_images, H_down, W_down, 3), dtype=np.float32)
    rgb_values_array = np.zeros((num_images, H_down, W_down, 3), dtype=np.float32)  # New array for RGB values
    points_3D_array = np.zeros((num_images, num_samples, H_down, W_down, 3), dtype=np.float32)
    
    for i, frame in enumerate(data["frames"][:num_images]):
        print(f"Processing Image {i+1}/{num_images} (downsampled with CUDA)")

        # Extract transformation matrix and convert to tensor
        transformation_matrix = torch.tensor(
            frame["transform_matrix"], device=device
        ).float()
        
        camera_position = transformation_matrix[:3, 3]
        rotation_matrix = transformation_matrix[:3, :3]

        # Create a meshgrid of pixel coordinates (downsampled)
        k, j = torch.meshgrid(
            torch.linspace(0, W - 1, W_down, device=device),
            torch.linspace(0, H - 1, H_down, device=device),
            indexing='xy'
        )

        # Normalize pixel coordinates
        k = (k - W / 2) / focal_length
        j = -(j - H / 2) / focal_length

        # Define ray directions in camera space
        ones = torch.ones_like(k)
        ray_directions_camera = torch.stack([k, j, -ones], dim=-1)

        # Convert ray directions to world space
        ray_directions_world = torch.matmul(ray_directions_camera, rotation_matrix.T)
        ray_directions_world /= torch.norm(ray_directions_world, dim=-1, keepdim=True)

        # Broadcast camera position
        ray_origins = camera_position.expand_as(ray_directions_world)

        # Load the RGB image and downsample
        image_path = os.path.join(current_dir, "nerf", "Data", "lego", frame["file_path"] + ".png")
        image = np.array(Image.open(image_path).resize((W_down, H_down))) / 255.0  # Normalize to [0, 1]
        
        # Store RGB values
        rgb_values_array[i] = image

        # Calculate 3D points with batched operations on GPU
        # Reshape for broadcasting
        origins_expanded = ray_origins.unsqueeze(0)  # [1, H, W, 3]
        directions_expanded = ray_directions_world.unsqueeze(0)  # [1, H, W, 3]
        depths_expanded = depth_samples_tensor.view(-1, 1, 1, 1)  # [D, 1, 1, 1]
        
        # Calculate points - this happens entirely on GPU
        points_3D = origins_expanded + depths_expanded * directions_expanded
        
        # Move results to CPU and store in the consolidated arrays
        ray_origins_array[i] = ray_origins.cpu().numpy()
        ray_directions_array[i] = ray_directions_world.cpu().numpy()
        points_3D_array[i] = points_3D.cpu().numpy()
        
        # Free GPU memory
        del points_3D, origins_expanded, directions_expanded
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save the consolidated arrays
    print("\nSaving consolidated data arrays...")
    np.save(os.path.join(output_dir, "ray_origins.npy"), ray_origins_array)
    np.save(os.path.join(output_dir, "ray_directions.npy"), ray_directions_array)
    np.save(os.path.join(output_dir, "rgb_values.npy"), rgb_values_array)  # Save RGB values
    np.save(os.path.join(output_dir, "points_3D.npy"), points_3D_array)
    
    # Create a flattened dataset for training
    print("\nCreating training dataset...")
    ray_origins_flat = ray_origins_array.reshape(-1, 3)
    ray_directions_flat = ray_directions_array.reshape(-1, 3)
    rgb_values_flat = rgb_values_array.reshape(-1, 3)
    
    # Combine into a single array for training
    training_data = np.concatenate([
        ray_origins_flat, 
        ray_directions_flat,
        rgb_values_flat
    ], axis=1)
    
    # Save training data
    np.save(os.path.join(output_dir, "training_data.npy"), training_data)
    
    print(f"Ray Origins: {ray_origins_array.shape} → saved as 'ray_origins.npy'")
    print(f"Ray Directions: {ray_directions_array.shape} → saved as 'ray_directions.npy'")
    print(f"RGB Values: {rgb_values_array.shape} → saved as 'rgb_values.npy'")
    print(f"3D Sample Points: {points_3D_array.shape} → saved as 'points_3D.npy'")
    print(f"Training Data: {training_data.shape} → saved as 'training_data.npy'")

# Choose which method to run
# Comment/uncomment the desired method

# For full resolution (all data in one array, but uses memory-mapped file for points_3D)
process_individual_images()  

# For downsampled resolution (all data in one array, fits in memory)
# process_with_downsampling(downsample_factor=4)

print("\n✅ Data Extraction Completed & Saved")
print(f"Output directory: {output_dir}")
print("\nCUDA Acceleration Summary:")
print(f"- CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"- GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"- GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
