import os
from pathlib import Path
import random
import argparse
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops, ImageEnhance, ImageColor
from tqdm import tqdm
from config import get_config

def create_receipt_collage(image_paths, canvas_size=(1600, 1200), receipt_count=None, 
                        bg_color=(255, 255, 255), realistic=False):
    """
    Create a simple collage of receipts laid out in a grid with minimal effects.
    For 0-receipt examples, generates an Australian tax document instead.
    Collages can be in either portrait or landscape orientation.
    
    Args:
        image_paths: List of paths to receipt images
        canvas_size: Size of the output canvas (width, height) - can be portrait or landscape
        receipt_count: Specific number of receipts to include (or None for random)
        bg_color: Background color of the canvas (always white, parameter kept for compatibility)
        realistic: Parameter kept for compatibility but not used
        
    Returns:
        collage_img: PIL Image of the collage or a tax document for 0-receipt cases
        actual_count: Number of receipts in the collage
        
    Note:
        When receipt_count is 0, this function returns an Australian tax document (ATO notice,
        PAYG summary, etc.) in portrait orientation to represent real-world tax documents
        that might be included in tax submissions but aren't receipts.
        
        Collages can be in either portrait or landscape orientation to mimic real-world
        photos taken with mobile phones, but tax documents are always in portrait orientation
        regardless of the collage orientation.
    """
    # Always create a plain white canvas
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    
    # If receipt_count is None, it should be set by the caller
    # based on the class distribution from config
    
    # If we need 0 receipts, generate an Australian tax document in portrait orientation
    if receipt_count == 0:
        try:
            # Import tax document generator if we need it
            from create_tax_documents import generate_tax_document
            
            # Generate a standard A4 portrait tax document (595x842 pixels at 72 DPI)
            tax_document = generate_tax_document(width=595, height=842)
            
            # Create a white background canvas to place the tax document on
            background = Image.new('RGB', canvas_size, color=(255, 255, 255))
            
            # Calculate how to fit tax document into canvas while preserving portrait orientation
            tax_width, tax_height = tax_document.size
            canvas_width, canvas_height = canvas_size
            
            # Always ensure tax document is in portrait orientation (height > width)
            # Even if the collage itself is landscape or portrait
            if tax_height < tax_width:
                # Rotate if document is not in portrait orientation
                tax_document = tax_document.transpose(Image.ROTATE_90)
                tax_width, tax_height = tax_height, tax_width
            
            # Calculate scale to fit while maintaining portrait orientation
            scale = min(canvas_height / tax_height, canvas_width / tax_width)
            new_height = int(tax_height * scale)
            new_width = int(tax_width * scale)
            tax_document_resized = tax_document.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the resized tax document on the canvas
            paste_x = (canvas_width - new_width) // 2
            paste_y = (canvas_height - new_height) // 2
            background.paste(tax_document_resized, (paste_x, paste_y))
            
            # Return the canvas with properly oriented tax document
            return background, 0
        except (ImportError, Exception) as e:
            print(f"Warning: Could not generate tax document: {e}")
            # Fall back to empty canvas if tax document generation fails
            return canvas, 0
    
    # Randomly select receipt images
    if receipt_count > len(image_paths):
        receipt_count = len(image_paths)
    selected_images = random.sample(image_paths, receipt_count)
    
    # More natural placement strategy
    # For realistic placement, we'll use:
    # 1. More variable grid with clustering
    # 2. Higher probability of overlap for higher receipt counts
    # 3. Proper z-ordering (later receipts on top)
    
    # Create a larger grid for better distribution
    grid_columns = 3  # Fixed number of columns for consistency
    grid_rows = 3     # Fixed number of rows for consistency
    
    # Calculate cell dimensions
    cell_width = canvas_size[0] // grid_columns
    cell_height = canvas_size[1] // grid_rows
    
    # We don't want overlap for the simplified version
    overlap_prob = 0.0  # No overlap
    overlap_allowed = False
    
    # Keep track of which grid cells are used
    grid_used = [[False for _ in range(grid_columns)] for _ in range(grid_rows)]
    
    # Simple uniform grid placement with minimal jitter
    def get_placement_cell():
        # If no receipts, doesn't matter
        if receipt_count == 0:
            return (0, 0)
        
        # Calculate cells needed for our receipts
        cells_needed = receipt_count
        
        # Calculate a grid that can fit all receipts
        # For any receipt count, create a uniform grid
        grid_size = math.ceil(math.sqrt(cells_needed))
        
        # Calculate optimal dimensions
        virtual_rows = min(grid_size, grid_rows)
        virtual_cols = math.ceil(cells_needed / virtual_rows)
        
        # Find which cells are already used
        placed_count = sum(1 for r in range(grid_rows) for c in range(grid_columns) if grid_used[r][c])
        
        # If we've already placed all receipts, find an unused cell
        if placed_count >= cells_needed:
            unused_cells = [(r, c) for r in range(grid_rows) for c in range(grid_columns) 
                           if not grid_used[r][c]]
            if unused_cells:
                cell = random.choice(unused_cells)
                if 0 <= cell[0] < grid_rows and 0 <= cell[1] < grid_columns:
                    grid_used[cell[0]][cell[1]] = True
                return cell
            else:
                # All positions used, return random position
                return (random.randint(0, grid_rows-1), random.randint(0, grid_columns-1))
        
        # Simple placement in a grid
        # Calculate row and column based on how many we've already placed
        row = placed_count // virtual_cols
        col = placed_count % virtual_cols
        
        # Scale to fit our actual grid
        row_scale = grid_rows / virtual_rows
        col_scale = grid_columns / virtual_cols
        
        # Calculate center position
        row = int(row * row_scale + row_scale / 2)
        col = int(col * col_scale + col_scale / 2)
        
        # Very minimal jitter (just enough for slight variation)
        row_jitter = random.randint(-1, 1)
        col_jitter = random.randint(-1, 1)
        
        row = max(0, min(grid_rows-1, row + row_jitter))
        col = max(0, min(grid_columns-1, col + col_jitter))
        
        # Mark this cell as used
        if 0 <= row < grid_rows and 0 <= col < grid_columns:
            grid_used[row][col] = True
        
        return (row, col)
    
    # Place each receipt
    actual_count = 0
    
    for img_path in selected_images:
        try:
            # Get an intelligent cell placement using our enhanced function
            cell = get_placement_cell()
            if cell is None:
                break  # No more cells available
                
            row, col = cell
            
            # Calculate the cell boundaries
            cell_x = col * cell_width
            cell_y = row * cell_height
            
            # Load and prepare the receipt
            receipt = Image.open(img_path).convert('RGB')
            
            # Simple background removal - just replace dark borders with white
            # Keep it simple - no fancy effects
            try:
                # Convert to RGB if not already
                receipt = receipt.convert('RGB')
                data = np.array(receipt)
                
                # Simple approach - replace very dark pixels with white (canvas background)
                r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
                
                # Create mask for dark borders/backgrounds (black or very dark)
                dark_pixels = (r < 50) & (g < 50) & (b < 50)
                
                # Replace dark pixels with white
                for channel in range(3):
                    data[:,:,channel][dark_pixels] = 255
                
                # Create the clean receipt with white background
                receipt = Image.fromarray(data)
                
                # Simple contrast adjustment to ensure text is readable
                enhancer = ImageEnhance.Contrast(receipt)
                receipt = enhancer.enhance(1.2)
                
            except Exception as e:
                print(f"Error processing receipt: {e}")
            
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
            
            # Simple rotation for slight variation
            # Keep it minimal - just a small rotation
            angle = random.uniform(-5, 5)  # Very slight rotation
            receipt = receipt.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
            
            # Simple cell-based placement with minimal jitter
            # Calculate the center of the cell
            cell_center_x = cell_x + cell_width // 2
            cell_center_y = cell_y + cell_height // 2
            
            # Add very small jitter (< 10% of cell size)
            max_jitter = min(cell_width, cell_height) // 10
            offset_x = random.randint(-max_jitter, max_jitter)
            offset_y = random.randint(-max_jitter, max_jitter)
            
            # Calculate final position - centered in cell with small jitter
            paste_x = cell_center_x - receipt.width // 2 + offset_x
            paste_y = cell_center_y - receipt.height // 2 + offset_y
            
            # Ensure receipt stays within canvas bounds
            paste_x = max(0, min(paste_x, canvas_size[0] - receipt.width))
            paste_y = max(0, min(paste_y, canvas_size[1] - receipt.height))
            
            # No shadows in simplified version
            
            # Paste the receipt onto the canvas
            canvas.paste(receipt, (paste_x, paste_y))
            actual_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return canvas, actual_count

def main():
    """
    Generate simple receipt collages for training vision transformer models.
    
    This script creates synthetic training data by placing receipts on a white background
    arranged in a grid pattern. It fully integrates with the config system to ensure 
    class distribution consistency across the entire codebase.
    
    Key features:
    - Simple white background for all collages
    - Grid-based placement with minimal jitter
    - Removes black borders from receipts
    - Minimal rotation for slight variation
    - No shadows or complex effects
    - No overlaps between receipts
    - 0-receipt examples use Australian tax documents instead of blank pages
    
    The class distribution can be specified via:
    1. Command line with --count_probs
    2. Configuration file with --config
    3. Default from the global config system
    
    Receipt counts are selected according to the probability distribution,
    and the actual distribution is reported at the end of generation.
    
    Note: For 0-receipt examples, the script will generate Australian tax documents
    (like ATO notices, PAYG summaries, etc.) in portrait orientation, simulating
    real-world tax submission scenarios.
    """
    parser = argparse.ArgumentParser(description="Create receipt collages for training vision transformer models")
    parser.add_argument("--input_dir", default="test_images", 
                        help="Directory containing receipt images")
    parser.add_argument("--output_dir", default="receipt_collages", 
                        help="Directory to save collage images")
    parser.add_argument("--num_collages", type=int, default=300,
                        help="Number of collages to create")
    parser.add_argument("--canvas_width", type=int, default=1600,
                        help="Width of the collage canvas (will be swapped for portrait orientation)")
    parser.add_argument("--canvas_height", type=int, default=1200,
                        help="Height of the collage canvas (will be swapped for portrait orientation)")
    parser.add_argument("--count_probs", type=str, 
                      help="Comma-separated probabilities for 0,1,2,3,4,5 receipts (overrides config)")
    parser.add_argument("--realistic", action="store_true", default=False,
                      help="Not used, kept for compatibility")
    parser.add_argument("--bg_color", type=str, default="255,255,255",
                      help="Background color (always white in simplified version, kept for compatibility)")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config) if args.config else None
    
    # Load configuration
    config = get_config()
    if config_path:
        if not config_path.exists():
            print(f"Warning: Configuration file not found: {config_path}")
        else:
            config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Parse probability distribution (command line args override config)
    if args.count_probs:
        try:
            # Parse probabilities from command line
            count_probs = [float(p) for p in args.count_probs.split(',')]
            
            # Normalize to ensure they sum to 1
            prob_sum = sum(count_probs)
            if prob_sum <= 0:
                raise ValueError("Probabilities must sum to a positive value")
            count_probs = [p / prob_sum for p in count_probs]
            print(f"Using receipt count distribution from command line: {count_probs}")
            
            # Update config with the new distribution
            if len(count_probs) == len(config.class_distribution):
                config.update_class_distribution(count_probs)
            else:
                print(f"Warning: Provided distribution has {len(count_probs)} values, " 
                      f"but configuration expects {len(config.class_distribution)}. Using provided values.")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Invalid probability format in command line: {e}")
            count_probs = config.class_distribution
            print(f"Using class distribution from config: {count_probs}")
    else:
        # Use distribution from config
        count_probs = config.class_distribution
        print(f"Using class distribution from config: {count_probs}")
        
    # Save to config file if specified
    if args.config and args.count_probs:
        config.save_to_file(config_path)
        print(f"Updated configuration saved to {config_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    # Use Path.rglob to collect image files
    for ext in image_extensions:
        image_files.extend([str(p) for p in input_dir.rglob(f"*{ext}")])
        image_files.extend([str(p) for p in input_dir.rglob(f"*{ext.upper()}")])  # Also match uppercase extensions
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Count distribution for verification
    count_distribution = {i: 0 for i in range(len(count_probs))}  # Match config distribution classes
    
    # Always use white background
    print("Using white background for all collages (255, 255, 255)")

    # Always use white background for simplicity
    white_bg = (255, 255, 255)  # Pure white
        
    # Create collages
    for i in tqdm(range(args.num_collages), desc="Creating collages"):
        # Randomly choose between portrait and landscape orientation
        is_portrait = random.choice([True, False])
        
        # If portrait, swap width and height
        if is_portrait:
            canvas_size = (args.canvas_height, args.canvas_width)  # Portrait (height > width)
        else:
            canvas_size = (args.canvas_width, args.canvas_height)  # Landscape (width > height)
        
        # Select receipt count based on probability distribution from config
        receipt_count = random.choices(list(range(len(count_probs))), weights=count_probs)[0]
        
        # Always use white background
        collage, actual_count = create_receipt_collage(
            image_files, canvas_size, receipt_count,
            bg_color=white_bg, realistic=False
        )
        
        # Track distribution
        count_distribution[actual_count] += 1
        
        # Save the collage using the original naming convention
        output_path = output_dir / f"collage_{i:03d}_{actual_count}_receipts.jpg"
        collage.save(output_path, "JPEG", quality=95)
    
    # Report final receipt count distribution
    print(f"\nActual receipt count distribution:")
    for count, freq in sorted(count_distribution.items()):
        percentage = freq / args.num_collages * 100
        print(f"  {count} receipts: {freq} collages ({percentage:.1f}%)")
    
    print(f"\nCreated {args.num_collages} collages in {output_dir}")

if __name__ == "__main__":
    main()