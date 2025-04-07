import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from receipt_processor import ReceiptProcessor
from model_factory import ModelFactory
from config import get_config
from device_utils import get_device

def process_image(model_path, image_path, enhance=None, config_path=None, model_type=None):
    """Process an image using the trained model."""
    # Load configuration
    config = get_config()
    if config_path:
        config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Get parameters from config if not explicitly provided
    if model_type is None:
        model_type = config.get_model_param("model_type", "swinv2")
    
    if enhance is None:
        enhance = True  # Default is to enhance images
    
    # Get the best available device
    device = get_device()
    
    # Load model based on model path
    print(f"Loading model from {model_path}...")
    
    try:
        # Load model using the factory with strict=False to allow for class count changes
        model = ModelFactory.load_model(
            model_path,
            strict=False,
            mode="eval",
            model_type=model_type
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
    
    # Format the output for the 3-class system
    if count == 2:
        count_display = "2+"
    else:
        count_display = str(count)
    
    print(f"Detected {count_display} receipts in the image.")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Visualize results
    try:
        # Open image with PIL (already in RGB format)
        img = np.array(Image.open(image_path))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Detected {count_display} receipts (Confidence: {confidence*100:.2f}%)")
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
    parser.add_argument("--model", default="models/receipt_counter_swinv2_best.pth",
                       help="Path to model (use the 'best' model, not the 'final' model)")
    parser.add_argument("--model-type", choices=["swinv2", "swinv2-large"],
                       help="Type of model to use (default from config)")
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
                      config_path=args.config,
                      model_type=args.model_type)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)