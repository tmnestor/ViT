import os
from pathlib import Path
import random
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import save_image

def create_receipt_collage(image_paths, canvas_size=(1600, 1200), max_count=5):
    """
    Create a collage of receipts using PyTorch transforms.
    
    Args:
        image_paths: List of paths to receipt images
        canvas_size: Size of the output canvas (width, height)
        max_count: Maximum number of receipts to include
        
    Returns:
        canvas_tensor: PyTorch tensor of the collage
        receipt_count: Number of receipts in the collage
    """
    # Create a white canvas
    canvas_tensor = torch.ones(3, canvas_size[1], canvas_size[0])
    
    # Randomly select how many receipts to include (0 to max_count)
    num_receipts = random.randint(0, min(max_count, len(image_paths)))
    selected_images = random.sample(image_paths, num_receipts)
    actual_count = 0
    
    for img_path in selected_images:
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Convert to tensor
            img_tensor = F.to_tensor(img)
            
            # Resize if needed (max dimension 1/3 of the canvas)
            max_dim = max(img_tensor.shape[1], img_tensor.shape[2])
            max_allowed = min(canvas_size) // 3
            
            if max_dim > max_allowed:
                scale_factor = max_allowed / max_dim
                new_height = int(img_tensor.shape[1] * scale_factor)
                new_width = int(img_tensor.shape[2] * scale_factor)
                img_tensor = F.resize(img_tensor, [new_height, new_width], antialias=True)
            
            # Apply random rotation
            angle = random.uniform(-20, 20)
            img_tensor = F.rotate(img_tensor, angle, expand=True)
            
            # Get dimensions of rotated image
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            
            # Generate random position
            max_x = max(0, canvas_size[0] - w)
            max_y = max(0, canvas_size[1] - h)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Place image on canvas
            try:
                # Create a mask for the receipt (non-white pixels)
                # Assuming receipts are mostly white with darker content
                brightness = img_tensor.mean(dim=0)
                mask = brightness < 0.95  # Threshold for determining receipt content
                
                # Place the receipt on the canvas
                canvas_region = canvas_tensor[:, y:y+h, x:x+w]
                canvas_region[:, mask] = img_tensor[:, mask]
                actual_count += 1
            except Exception as e:
                print(f"Error placing image: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return canvas_tensor, actual_count

def main():
    parser = argparse.ArgumentParser(description="Create receipt collages using PyTorch")
    parser.add_argument("--input_dir", default="test_images", 
                        help="Directory containing receipt images")
    parser.add_argument("--output_dir", default="collages", 
                        help="Directory to save collage images")
    parser.add_argument("--num_collages", type=int, default=100,
                        help="Number of collages to create")
    parser.add_argument("--canvas_width", type=int, default=1600,
                        help="Width of the collage canvas")
    parser.add_argument("--canvas_height", type=int, default=1200,
                        help="Height of the collage canvas")
    parser.add_argument("--max_images", type=int, default=5,
                        help="Maximum number of images per collage")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create collages
    for i in tqdm(range(args.num_collages), desc="Creating collages"):
        canvas_size = (args.canvas_width, args.canvas_height)
        collage, count = create_receipt_collage(image_files, canvas_size, args.max_images)
        
        # Save the collage
        output_path = os.path.join(args.output_dir, f"collage_{i:03d}_{count}_receipts.jpg")
        save_image(collage, output_path)
    
    print(f"Created {args.num_collages} collages in {args.output_dir}")

if __name__ == "__main__":
    main()