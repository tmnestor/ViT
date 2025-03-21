import os
import argparse
import matplotlib.pyplot as plt
import random
from glob import glob
from PIL import Image
import torch
from receipt_processor import ReceiptProcessor
from model_factory import ModelFactory
from device_utils import get_device

def show_sample_images(image_dir, num_samples=4):
    """Display a grid of sample images from the test directory."""
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(image_files))
    samples = random.sample(image_files, num_samples)
    
    # Display images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(samples):
        if i >= num_samples:
            break
            
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(img_path))
            axes[i].axis('off')
        except Exception as e:
            print(f"Error displaying {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()
    
    print(f"Displayed {num_samples} sample images from {image_dir}")
    print(f"Total images available: {len(image_files)}")
    return image_files

def process_multiple_images(model_path, image_dir, num_samples=4):
    """Process multiple images from the test directory using the trained model."""
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(image_files))
    samples = random.sample(image_files, num_samples)
    
    # Load model
    device = get_device()
    try:
        # Determine if it's a ViT or Swin model based on filename
        model_type = "vit" if "vit" in model_path.lower() else "swin"
        model = ModelFactory.load_model(
            model_path, 
            model_type=model_type, 
            strict=True, 
            mode="eval"
        ).to(device)
        processor = ReceiptProcessor()
        print(f"Successfully loaded {model_type.upper()} model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model loading failed. Please ensure the model file exists and is in the correct format.")
        return
    
    # Process images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(samples):
        if i >= num_samples:
            break
            
        try:
            # Preprocess the image
            img_tensor = processor.preprocess_image(img_path).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                # Get logits from model outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get class prediction (0-5)
                probs = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            # The predicted class is directly the count (0-5)
            count = predicted_class
            
            # Display results
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Detected {count} receipts\nConfidence: {confidence:.2%}")
            axes[i].axis('off')
            
            print(f"Image: {os.path.basename(img_path)}")
            print(f"  Detected {count} receipts")
            print(f"  Confidence: {confidence:.2%}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                         ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig("batch_results.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt Counter Demo with Test Images")
    parser.add_argument("--image_dir", default="test_images", 
                       help="Directory containing test images")
    parser.add_argument("--model", default="receipt_counter_swin_tiny.pth",
                       help="Path to trained model")
    parser.add_argument("--samples", type=int, default=4,
                       help="Number of sample images to process")
    parser.add_argument("--mode", choices=["show", "process"], default="show",
                       help="Show sample images or process them with model")
    
    args = parser.parse_args()
    
    if args.mode == "show":
        image_files = show_sample_images(args.image_dir, args.samples)
        if image_files:
            print("\nTo process these images, run:")
            print(f"python test_images_demo.py --mode process --model {args.model}")
    else:
        process_multiple_images(args.model, args.image_dir, args.samples)