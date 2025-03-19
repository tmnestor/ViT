import requests
import json
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from receipt_processor import ReceiptProcessor
from receipt_counter import ReceiptCounter
import torch

# Try to import OpenCV, but provide helpful error if not installed
try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is not installed. Please install it with:")
    print("pip install opencv-python")
    print("\nAlternatively, you can run without image enhancement:")
    print("pip install opencv-python-headless")
    sys.exit(1)

def process_local_image(model_path, image_path, enhance=True):
    """Process a local image using the trained model."""
    # Determine device (support CUDA, MPS, and CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = ReceiptCounter.load(model_path).to(device)
    processor = ReceiptProcessor()
    
    # Enhance the scan if requested
    if enhance:
        try:
            processor.enhance_scan_quality(image_path, "enhanced_scan.jpg")
            print("Enhanced scan saved as 'enhanced_scan.jpg'")
        except Exception as e:
            print(f"Warning: Could not enhance image: {e}")
    
    # Preprocess the image
    try:
        img_tensor = processor.preprocess_image(image_path).to(device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)
    
    # Make prediction
    with torch.no_grad():
        predicted_count = model.predict(img_tensor)
    
    # Round to nearest integer
    count = round(predicted_count)
    
    print(f"Detected {count} receipts in the image.")
    print(f"Raw prediction: {predicted_count:.2f}")
    
    # Visualize results
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Detected {count} receipts")
        plt.axis('off')
        plt.savefig("result.jpg")
        print("Visualization saved as 'result.jpg'")
        plt.show()
    except Exception as e:
        print(f"Warning: Could not visualize results: {e}")
    
    return count

def process_api_request(image_path, api_url="http://localhost:5000/count_receipts"):
    """Send image to API endpoint for processing."""
    with open(image_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(api_url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detected {result['receipt_count']} receipts")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Raw prediction: {result['raw_prediction']:.2f}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt Counter Demo")
    parser.add_argument("--image", required=True, help="Path to scanned image")
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                       help="Process locally or via API")
    parser.add_argument("--model", default="models/receipt_counter_best.pth",
                       help="Path to model (for local mode)")
    parser.add_argument("--api_url", default="http://localhost:5000/count_receipts",
                       help="API URL (for API mode)")
    parser.add_argument("--no-enhance", action="store_true",
                       help="Skip image enhancement (use if OpenCV has issues)")
    
    args = parser.parse_args()
    
    # Verify that the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    # Verify that the model exists for local mode
    if args.mode == "local" and not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Available models:")
        model_dir = os.path.dirname(args.model) or "."
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".pth"):
                    print(f" - {os.path.join(model_dir, file)}")
        sys.exit(1)
    
    try:
        if args.mode == "local":
            process_local_image(args.model, args.image, enhance=not args.no_enhance)
        else:
            process_api_request(args.image, args.api_url)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)