import os
import numpy as np
import glob
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import argparse
import json

def add_white_background(image):
    """
    Add a white background to an image with alpha channel.
    If the image doesn't have an alpha channel, return it unchanged.
    """
    if image.shape[-1] == 4:  # Has alpha channel
        # Extract RGB and alpha
        rgb = image[..., :3]
        alpha = image[..., 3:4]
        
        # White background [1,1,1]
        white_bg = np.ones_like(rgb)
        
        # Composite image over white background
        composited = rgb * alpha + white_bg * (1 - alpha)
        
        return composited
    else:
        return image  # No alpha channel, return unchanged

def preprocess_ground_truth(gt_dir, output_dir, target_size=(400, 400)):
    """
    Process all ground truth images in gt_dir by:
    1. Adding white background
    2. Downsampling to target_size
    And save them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(os.path.join(gt_dir, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} ground truth images...")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            # Load image with alpha channel if present
            img = Image.open(img_path).convert("RGBA")
            
            # Resize to target size
            img_resized = img.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array for white background compositing
            img_array = np.array(img_resized) / 255.0
            
            # Add white background
            processed_img = add_white_background(img_array)
            
            # Save processed image
            output_path = os.path.join(output_dir, filename)
            Image.fromarray((processed_img * 255).astype(np.uint8)).convert("RGB").save(output_path)
            
            print(f"Processed: {filename} -> {target_size}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Finished processing ground truth images. Saved to {output_dir}")
    return output_dir

def calculate_metrics(generated_dir, gt_dir, target_size=(400, 400)):
    """
    Calculate PSNR and SSIM metrics between generated images and ground truth images.
    Handles filename mapping between r_X.png and img_X.png formats.
    
    Args:
        generated_dir: Directory containing generated images (.png or .npy)
        gt_dir: Directory containing ground truth images
        target_size: Size to resize images to (width, height)
    
    Returns:
        Average PSNR and SSIM values
    """
    psnr_values = []
    ssim_values = []
    
    # Get all generated images (.png or .npy files)
    generated_files = sorted(glob.glob(os.path.join(generated_dir, 'img_*.png')))
    if not generated_files:
        generated_files = sorted(glob.glob(os.path.join(generated_dir, '*.npy')))
    
    print(f"Found {len(generated_files)} generated images")
    
    # Create a mapping of indices to ground truth files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, 'r_*.png')))
    gt_indices = {}
    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        match = re.search(r'r_(\d+).png', base_name, re.IGNORECASE)
        if match:
            index = int(match.group(1))
            gt_indices[index] = gt_file
    
    print(f"Found {len(gt_indices)} ground truth images")
    
    # Process each generated image
    for gen_path in generated_files:
        base_name = os.path.basename(gen_path)
        
        # Extract index from filename (img_X.png or similar)
        match = re.search(r'img_(\d+).png', base_name, re.IGNORECASE)
        if not match and gen_path.endswith('.npy'):
            match = re.search(r'(\d+).npy', base_name)
        
        if not match:
            print(f"Warning: Couldn't extract index from {base_name}, skipping")
            continue
        
        index = int(match.group(1))
        
        # Find corresponding ground truth file
        if index not in gt_indices:
            print(f"Warning: No ground truth file found for index {index}, skipping")
            continue
        
        gt_path = gt_indices[index]
        
        try:
            # Load generated image
            if gen_path.endswith('.npy'):
                gen_img = np.load(gen_path)
                # Ensure values are in [0, 1] range
                gen_img = np.clip(gen_img, 0, 1)
                
                # Resize if necessary
                if gen_img.shape[0] != target_size[1] or gen_img.shape[1] != target_size[0]:
                    gen_img_pil = Image.fromarray((gen_img * 255).astype(np.uint8))
                    gen_img_pil = gen_img_pil.resize(target_size, Image.LANCZOS)
                    gen_img = np.array(gen_img_pil) / 255.0
                
                # If generated image has alpha channel, composite it onto white background
                gen_img = add_white_background(gen_img)
            else:  # PNG
                gen_img_pil = Image.open(gen_path).convert("RGBA")
                
                # Resize if necessary
                if gen_img_pil.size != target_size:
                    gen_img_pil = gen_img_pil.resize(target_size, Image.LANCZOS)
                
                gen_img = np.array(gen_img_pil) / 255.0
                gen_img = add_white_background(gen_img)
            
            # Load ground truth image
            gt_img_pil = Image.open(gt_path).convert("RGBA")
            
            # Resize to target size
            if gt_img_pil.size != target_size:
                gt_img_pil = gt_img_pil.resize(target_size, Image.LANCZOS)
            
            gt_img = np.array(gt_img_pil) / 255.0
            gt_img = add_white_background(gt_img)
            
            # Convert to RGB for metrics calculation
            gen_img_rgb = gen_img[..., :3] if gen_img.shape[-1] >= 3 else gen_img
            gt_img_rgb = gt_img[..., :3] if gt_img.shape[-1] >= 3 else gt_img
            
            # Calculate metrics
            # For PSNR, when images are in [0,1] range, we set data_range=1
            current_psnr = psnr(gt_img_rgb, gen_img_rgb, data_range=1.0)
            
            # For SSIM, specify data_range and also use multichannel parameter
            current_ssim = ssim(
                gt_img_rgb, 
                gen_img_rgb, 
                data_range=1.0,
                channel_axis=-1 if len(gt_img_rgb.shape) > 2 else None
            )
            
            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)
            
            print(f"Image {index}: PSNR = {current_psnr:.4f} dB, SSIM = {current_ssim:.4f}")
            
        except Exception as e:
            print(f"Error processing index {index}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Calculate and return averages
    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        return avg_psnr, avg_ssim, psnr_values, ssim_values
    else:
        print("No images could be processed. Check paths and formats.")
        return None, None, [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM between generated and ground truth images')
    parser.add_argument('--gen', default='/home/sviswasam/cv/nerf/novel_views/', help='Directory containing generated images')
    parser.add_argument('--gt', default='/home/sviswasam/cv/nerf/Data/lego/test/', help='Directory containing ground truth images')
    parser.add_argument('--output', default='metrics_results.txt', help='Output file for metrics summary')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess ground truth images with white background and downsampling')
    parser.add_argument('--target_size', default='400,400', help='Target size for downsampling (width,height)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed debug information')
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    print(f"Calculating metrics...")
    print(f"Generated images: {args.gen}")
    print(f"Ground truth: {args.gt}")
    print(f"Target size: {target_size}")
    
    # Show debug info if requested
    if args.verbose:
        # Show what files exist in the directories
        print("\nGenerated directory contents:")
        gen_files = sorted(os.listdir(args.gen))
        for f in gen_files[:10]:  # Show first 10 files
            print(f"  {f}")
        if len(gen_files) > 10:
            print(f"  ... and {len(gen_files) - 10} more files")
            
        print("\nGround truth directory contents:")
        gt_files = sorted(os.listdir(args.gt))
        for f in gt_files[:10]:  # Show first 10 files
            print(f"  {f}")
        if len(gt_files) > 10:
            print(f"  ... and {len(gt_files) - 10} more files")
    
    # Preprocess ground truth images if requested
    if args.preprocess:
        print("Preprocessing ground truth images with white background and downsampling...")
        processed_gt_dir = os.path.join(os.path.dirname(args.gt), "processed_gt")
        gt_dir = preprocess_ground_truth(args.gt, processed_gt_dir, target_size)
    else:
        gt_dir = args.gt
    
    avg_psnr, avg_ssim, psnr_values, ssim_values = calculate_metrics(args.gen, gt_dir, target_size)
    
    if avg_psnr is not None:
        print("\nResults Summary:")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save to file
        with open(args.output, 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
            f.write("Individual image metrics:\n")
            
            for i, (p, s) in enumerate(zip(psnr_values, ssim_values)):
                f.write(f"Image {i}: PSNR = {p:.4f} dB, SSIM = {s:.4f}\n")
        
        print(f"\nMetrics saved to {args.output}")


    #     parser.add_argument('--gen', default='/home/sviswasam/cv/nerf/novel_views/', help='Directory containing generated images')
    # parser.add_argument('--gt', default='/home/sviswasam/cv/nerf/Data/lego/test/', help='Directory containing ground truth images')
