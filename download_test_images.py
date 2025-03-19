import os
import sys
import urllib.request
import zipfile
import shutil
import re
from tqdm import tqdm

# Create test_images directory if it doesn't exist
os.makedirs("test_images", exist_ok=True)

# URL information for the SRD dataset
SRD_DOWNLOAD_PAGE = "https://expressexpense.com/blog/research-free-receipt-image-dataset-for-receipt-ocr/"
ZIP_PATH = "SRD_dataset.zip"
EXTRACT_DIR = "SRD_temp"

# Download progress bar class
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def clean_filename(filename):
    """Clean up a filename and convert to sample_receipt_X.jpg format."""
    # Extract numeric parts of the filename
    matches = re.findall(r'\d+', filename)
    if matches:
        # Use the first numeric sequence found
        number = matches[0]
        return f"sample_receipt_{number}.jpg"
    else:
        # If no number found, generate one from hash
        import hashlib
        hash_num = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10000
        return f"sample_receipt_{hash_num}.jpg"

def extract_rename_images():
    """Extract and rename images from the SRD dataset."""
    if not os.path.exists(ZIP_PATH):
        print(f"Zip file '{ZIP_PATH}' not found. Please download the dataset manually.")
        return False
    
    # Create temporary directory for extraction
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    try:
        # Extract zip file
        print("Extracting zip file...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        
        # Find all images in the extracted folder (recursively)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        print("Finding images...")
        for root, _, files in os.walk(EXTRACT_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # Copy and rename the images
        print(f"Copying {len(image_files)} images to test_images/...")
        for i, img_path in enumerate(tqdm(image_files)):
            # Generate target filename
            filename = os.path.basename(img_path)
            new_filename = clean_filename(filename)
            
            # If file already exists, add a suffix
            if os.path.exists(os.path.join("test_images", new_filename)):
                base, ext = os.path.splitext(new_filename)
                new_filename = f"{base}_{i}{ext}"
            
            # Copy file
            shutil.copy2(img_path, os.path.join("test_images", new_filename))
        
        return True
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False
    finally:
        # Clean up: remove the temp directory
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)

def create_placeholder_images():
    """Create placeholder receipt images if no real images are available."""
    print("Creating sample placeholder receipt images...")
    try:
        from PIL import Image, ImageDraw
        
        for i in range(1, 6):
            # Create a white image
            img = Image.new('RGB', (300, 500), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Add some text
            draw.text((10, 10), f"Sample Receipt #{i}", fill=(0, 0, 0))
            draw.text((10, 30), "----------------------", fill=(0, 0, 0))
            draw.text((10, 50), "Item 1          $10.99", fill=(0, 0, 0))
            draw.text((10, 70), "Item 2          $24.50", fill=(0, 0, 0))
            draw.text((10, 90), "Item 3           $7.25", fill=(0, 0, 0))
            draw.text((10, 110), "----------------------", fill=(0, 0, 0))
            draw.text((10, 130), "Subtotal        $42.74", fill=(0, 0, 0))
            draw.text((10, 150), "Tax              $3.42", fill=(0, 0, 0))
            draw.text((10, 170), "Total          $46.16", fill=(0, 0, 0))
            
            # Save the image
            img.save(f"test_images/sample_receipt_{i}.png")
        
        print("Created 5 placeholder receipt images in test_images/ directory.")
    except Exception as e:
        print(f"Error creating placeholder images: {e}")

def main():
    print("SRD Receipt Dataset Processor")
    print("-----------------------------")
    
    # Check if the zip file already exists
    if os.path.exists(ZIP_PATH):
        print(f"Found existing SRD dataset zip file: {ZIP_PATH}")
        
        # Process the existing zip file
        if extract_rename_images():
            print("Successfully processed SRD dataset images.")
        else:
            print("Failed to process images.")
            create_placeholder_images()
        
        # Clean up the zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            print(f"Removed zip file: {ZIP_PATH}")
    else:
        print("SRD dataset zip file not found.")
        print("\nThe SRD dataset requires manual download due to access restrictions.")
        print(f"Please visit: {SRD_DOWNLOAD_PAGE}")
        print("\nDownload instructions:")
        print("1. Fill out the form on the website")
        print("2. Download the SRD.zip file")
        print(f"3. Save it as '{ZIP_PATH}' in this directory")
        print("4. Run this script again to process the images")
        
        # Create placeholder images for now
        create_placeholder_images()
    
    # Count the number of images
    image_count = len([f for f in os.listdir("test_images") 
                    if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    print(f"\nTotal receipt images in test_images directory: {image_count}")
    print("\nTo use these images with demo.py, run:")
    print("python demo.py --image test_images/[image_filename] --mode local --model [model_path]")
    
    # If we only have the placeholder images, suggest alternative datasets
    if image_count <= 5:
        print("\nNOTE: You currently only have placeholder images.")
        print("For real receipt images, either:")
        print("1. Download the SRD dataset as instructed above, or")
        print("2. Use the create_realistic_collages.py script to generate synthetic data")

if __name__ == "__main__":
    main()