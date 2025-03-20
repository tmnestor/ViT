import os
import random
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
from config import get_config

def create_receipt_collage(image_paths, canvas_size=(1600, 1200), receipt_count=None, 
                        bg_color=(245, 245, 245), realistic=True):
    """
    Create a collage of receipts with realistic appearance.
    
    Args:
        image_paths: List of paths to receipt images
        canvas_size: Size of the output canvas (width, height)
        receipt_count: Specific number of receipts to include (or None for random)
        bg_color: Background color of the canvas
        realistic: If True, make the receipts blend with the background
        
    Returns:
        collage_img: PIL Image of the collage
        actual_count: Number of receipts in the collage
    """
    # Create a canvas with background color (slightly off-white for realism)
    canvas = Image.new('RGB', canvas_size, color=bg_color)
    
    # Determine how many receipts to include
    if receipt_count is None:
        receipt_count = random.randint(0, 5)
    
    # If we need 0 receipts, just return the canvas
    if receipt_count == 0:
        return canvas, 0
    
    # Randomly select receipt images
    if receipt_count > len(image_paths):
        receipt_count = len(image_paths)
    selected_images = random.sample(image_paths, receipt_count)
    
    # Define a grid for placing receipts
    # For up to 5 receipts, we'll use a 3x2 grid (6 cells)
    grid_columns = 3
    grid_rows = 2
    
    # Calculate cell dimensions
    cell_width = canvas_size[0] // grid_columns
    cell_height = canvas_size[1] // grid_rows
    
    # Keep track of which grid cells are used
    grid_used = [[False for _ in range(grid_columns)] for _ in range(grid_rows)]
    
    # Function to get unused grid cell
    def get_unused_cell():
        unused_cells = [(r, c) for r in range(grid_rows) for c in range(grid_columns) 
                        if not grid_used[r][c]]
        if not unused_cells:
            return None
        return random.choice(unused_cells)
    
    # Place each receipt
    actual_count = 0
    
    for img_path in selected_images:
        try:
            # Get an unused cell
            cell = get_unused_cell()
            if cell is None:
                break  # No more cells available
                
            row, col = cell
            grid_used[row][col] = True
            
            # Calculate the cell boundaries
            cell_x = col * cell_width
            cell_y = row * cell_height
            
            # Load and prepare the receipt
            receipt = Image.open(img_path).convert('RGB')
            
            # Make receipts look like white paper on a colored background
            if realistic:
                try:
                    # Detect the dark rectangular background of the receipt
                    # Create a mask where dark pixels are white and light pixels are black
                    dark_mask = Image.new('L', receipt.size, 0)  # Start with black
                    
                    # Find dark pixels (the rectangle surrounding the receipt)
                    for x in range(receipt.width):
                        for y in range(receipt.height):
                            pixel = receipt.getpixel((x, y))
                            avg = sum(pixel) // 3
                            if avg < 100:  # Very dark pixels - the rectangle boundary
                                dark_mask.putpixel((x, y), 255)  # Mark as white in mask
                    
                    # Use the dark_mask to replace the black rectangle with background color
                    receipt_realistic = receipt.copy()
                    for x in range(receipt.width):
                        for y in range(receipt.height):
                            if dark_mask.getpixel((x, y)) > 200:  # It's part of the black rectangle
                                receipt_realistic.putpixel((x, y), bg_color)  # Replace with background color
                    
                    receipt = receipt_realistic
                except Exception as e:
                    print(f"Error processing receipt for realism: {e}")
            
            # Resize to fit within the cell (with margin)
            margin = 20  # pixels margin
            max_width = cell_width - 2 * margin
            max_height = cell_height - 2 * margin
            
            # Calculate scale to fit within max dimensions
            receipt_width, receipt_height = receipt.size
            width_scale = max_width / receipt_width
            height_scale = max_height / receipt_height
            scale = min(width_scale, height_scale)
            
            # If scale > 1, don't enlarge the image
            if scale > 1:
                scale = 1
                
            new_width = int(receipt_width * scale)
            new_height = int(receipt_height * scale)
            receipt = receipt.resize((new_width, new_height), Image.LANCZOS)
            
            # Apply a slight rotation (±10 degrees)
            angle = random.uniform(-10, 10)
            receipt = receipt.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=bg_color)
            
            # Calculate random position within the cell (centered with slight variation)
            cell_center_x = cell_x + cell_width // 2
            cell_center_y = cell_y + cell_height // 2
            
            # Add slight random offset for natural look 
            max_offset = min(20, (cell_width - receipt.width) // 2, (cell_height - receipt.height) // 2)
            max_offset = max(0, max_offset)  # Ensure it's not negative
            offset_x = random.randint(-max_offset, max_offset) if max_offset > 0 else 0
            offset_y = random.randint(-max_offset, max_offset) if max_offset > 0 else 0
            
            # Calculate final position
            paste_x = cell_center_x - receipt.width // 2 + offset_x
            paste_y = cell_center_y - receipt.height // 2 + offset_y
            
            # Add subtle shadow to create depth
            if realistic:
                shadow_offset = 3
                shadow_strength = 30
                shadow_color = (max(0, bg_color[0]-shadow_strength),
                               max(0, bg_color[1]-shadow_strength),
                               max(0, bg_color[2]-shadow_strength))
                
                # Create shadow
                shadow = Image.new('RGB', receipt.size, shadow_color)
                
                # Paste shadow with offset
                canvas.paste(shadow, (paste_x+shadow_offset, paste_y+shadow_offset))
            
            # Paste the receipt onto the canvas
            canvas.paste(receipt, (paste_x, paste_y))
            actual_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return canvas, actual_count

def main():
    parser = argparse.ArgumentParser(description="Create receipt collages for training")
    parser.add_argument("--input_dir", default="test_images", 
                        help="Directory containing receipt images")
    parser.add_argument("--output_dir", default="receipt_collages", 
                        help="Directory to save collage images")
    parser.add_argument("--num_collages", type=int, default=300,
                        help="Number of collages to create")
    parser.add_argument("--canvas_width", type=int, default=1600,
                        help="Width of the collage canvas")
    parser.add_argument("--canvas_height", type=int, default=1200,
                        help="Height of the collage canvas")
    parser.add_argument("--count_probs", type=str, 
                      help="Comma-separated probabilities for 0,1,2,3,4,5 receipts (overrides config)")
    parser.add_argument("--realistic", action="store_true", default=True,
                      help="Make receipts blend with background for more realistic appearance")
    parser.add_argument("--bg_color", type=str, default="245,245,245",
                      help="Background color in RGB format (e.g., '245,245,245' for light gray)")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    if args.config:
        if not os.path.exists(args.config):
            print(f"Warning: Configuration file not found: {args.config}")
        else:
            config.load_from_file(args.config)
            print(f"Loaded configuration from {args.config}")
    
    # Parse probability distribution (command line args override config)
    if args.count_probs:
        try:
            count_probs = [float(p) for p in args.count_probs.split(',')]
            # Normalize to ensure they sum to 1
            prob_sum = sum(count_probs)
            if prob_sum <= 0:
                raise ValueError("Probabilities must sum to a positive value")
            count_probs = [p / prob_sum for p in count_probs]
            print(f"Using receipt count distribution from command line: {count_probs}")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Invalid probability format in command line: {e}")
            count_probs = config.class_distribution
            print(f"Using class distribution from config: {count_probs}")
    else:
        # Use distribution from config if not specified on command line
        count_probs = config.class_distribution
        print(f"Using class distribution from config: {count_probs}")
        
    # Optionally, save the used distribution back to the configuration
    if not args.count_probs:
        config.update_class_distribution(count_probs)
        if args.config:
            config.save_to_file(args.config)
            print(f"Updated configuration saved to {args.config}")
    
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
    
    # Count distribution for verification
    count_distribution = {i: 0 for i in range(6)}  # 0-5 receipts
    
    # Parse background color
    try:
        bg_color = tuple(int(c) for c in args.bg_color.split(','))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            print(f"Warning: Invalid bg_color format, using default color")
            bg_color = (245, 245, 245)
    except ValueError:
        print(f"Warning: Invalid bg_color format, using default color")
        bg_color = (245, 245, 245)

    # Create collages
    for i in tqdm(range(args.num_collages), desc="Creating collages"):
        canvas_size = (args.canvas_width, args.canvas_height)
        
        # Select receipt count based on probability distribution
        receipt_count = random.choices(list(range(len(count_probs))), weights=count_probs)[0]
        
        collage, actual_count = create_receipt_collage(
            image_files, canvas_size, receipt_count,
            bg_color=bg_color, realistic=args.realistic
        )
        
        # Track distribution
        count_distribution[actual_count] += 1
        
        # Save the collage
        output_path = os.path.join(args.output_dir, f"collage_{i:03d}_{actual_count}_receipts.jpg")
        collage.save(output_path, "JPEG", quality=95)
    
    # Report final receipt count distribution
    print(f"\nActual receipt count distribution:")
    for count, freq in sorted(count_distribution.items()):
        percentage = freq / args.num_collages * 100
        print(f"  {count} receipts: {freq} collages ({percentage:.1f}%)")
    
    print(f"\nCreated {args.num_collages} collages in {args.output_dir}")

if __name__ == "__main__":
    main()