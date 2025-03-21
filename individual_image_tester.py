import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from receipt_processor import ReceiptProcessor
from model_factory import ModelFactory
from config import get_config
from device_utils import get_device

# Try to import OpenCV, but provide helpful error if not installed
try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is not installed. Please install it with:")
    print("pip install opencv-python")
    print("\nAlternatively, you can run without image enhancement:")
    print("pip install opencv-python-headless")
    sys.exit(1)

def process_image(model_path, image_path, enhance=True, config_path=None):
    """Process an image using the trained model."""
    # Load configuration
    config = get_config()
    if config_path:
        config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Get the best available device
    device = get_device()
    
    # Load model based on model path
    print(f"Loading model from {model_path}...")
    
    # Determine if it's a ViT or Swin model based on filename
    model_type = "vit" if "vit" in model_path.lower() else "swin"
    
    try:
        # Load model using the factory with strict=True
        model = ModelFactory.load_model(
            model_path, 
            model_type=model_type, 
            strict=True,
            mode="eval"
        )
        model = model.to(device)
        print("Successfully loaded model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
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
        outputs = model(img_tensor)
        
        # Get logits from HuggingFace model output
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        # Get raw logits and probabilities 
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Apply calibration to counter the bias from class weighting during training
        # This helps correct the tendency to over-predict minority classes (2-5)
        class_prior = config.get_class_prior_tensor(device)
        
        # Apply stronger calibration to higher count classes (which are over-predicted)
        # These calibration factors counteract the inverse weighting from training
        calibration_factors = config.get_calibration_tensor(device)
        
        # Apply calibration
        calibrated_probs = probs[0] * calibration_factors * class_prior
        # Re-normalize to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum()
        
        # Get both raw and calibrated predictions
        raw_predicted_class = torch.argmax(probs[0], dim=0).item()
        calibrated_predicted_class = torch.argmax(calibrated_probs, dim=0).item()
        
        # Get confidences
        raw_confidence = probs[0, raw_predicted_class].item()
        calibrated_confidence = calibrated_probs[calibrated_predicted_class].item()
        
        # Print both raw and calibrated predictions
        print(f"\nRaw prediction: {raw_predicted_class} (Confidence: {raw_confidence*100:.2f}%)")
        print(f"Calibrated prediction: {calibrated_predicted_class} (Confidence: {calibrated_confidence*100:.2f}%)")
        print("\nClass probabilities:")
        num_classes = len(config.class_distribution)
        for i in range(num_classes):
            print(f"  Class {i}: Raw {probs[0, i].item()*100:.2f}%, Calibrated {calibrated_probs[i].item()*100:.2f}%")
        
        # Use calibrated prediction as final count
        count = calibrated_predicted_class
        confidence = calibrated_confidence
    
    print(f"Detected {count} receipts in the image.")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Visualize results
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Detected {count} receipts (Confidence: {confidence*100:.2f}%)")
        plt.axis('off')
        plt.savefig("result.jpg")
        print("Visualization saved as 'result.jpg'")
        plt.show()
    except Exception as e:
        print(f"Warning: Could not visualize results: {e}")
    
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt Counter Demo")
    parser.add_argument("--image", required=True, help="Path to scanned image")
    parser.add_argument("--model", default="models/receipt_counter_swin_best.pth",
                       help="Path to model")
    parser.add_argument("--no-enhance", action="store_true",
                       help="Skip image enhancement (use if OpenCV has issues)")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Verify that the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
        
    # Verify that the model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Available models:")
        model_dir = os.path.dirname(args.model) or "."
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".pth"):
                    print(f" - {os.path.join(model_dir, file)}")
        sys.exit(1)
    
    # Verify config file exists if specified
    if args.config and not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        process_image(args.model, args.image, 
                      enhance=not args.no_enhance, 
                      config_path=args.config)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)