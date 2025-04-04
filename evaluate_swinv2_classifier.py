"""
SwinV2 model evaluation script that uses the unified evaluation module.
"""

import argparse
import sys
from pathlib import Path
from evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SwinV2-Tiny classifier model")
    parser.add_argument("--model", required=True, 
                       help="Path to the trained model")
    parser.add_argument("--test_csv", required=True,
                       help="Path to CSV file containing test data")
    parser.add_argument("--test_dir", required=True,
                       help="Directory containing test images")
    parser.add_argument("--batch_size", type=int,
                       help="Batch size (uses config default if not specified)")
    parser.add_argument("--model_variant", default="best",
                       choices=["best", "best_bacc", "best_f1", "final"],
                       help="Model variant to evaluate (default: best)")
    parser.add_argument("--model_type", choices=["swinv2", "swinv2-large"],
                       help="Type of SwinV2 model to use (default from config)")
    parser.add_argument("--output_dir", default="evaluation_swinv2",
                       help="Directory to save evaluation results")
    parser.add_argument("--config", 
                       help="Path to configuration JSON file")
    parser.add_argument("--binary", action="store_true",
                       help="Evaluate as binary classification (multiple receipts or not)")
    parser.add_argument("--no-calibration", action="store_true",
                       help="Disable Bayesian calibration for predictions")
    
    args = parser.parse_args()
    
    # Get configuration to determine model type if not provided
    from config import get_config
    config = get_config()
    
    # Get model_type from args or config
    model_type = args.model_type if args.model_type is not None else config.get_model_param("model_type", "swinv2")
    
    # If a model directory is provided instead of a specific model file,
    # construct the path using the model_variant and model_type arguments
    model_path = Path(args.model)
    if model_path.is_dir():
        # Define variant map based on model type
        if model_type == "swinv2-large":
            variant_map = {
                "best": f"receipt_counter_{model_type}_best.pth",
                "best_bacc": f"receipt_counter_{model_type}_best_bacc.pth",
                "best_f1": f"receipt_counter_{model_type}_best_f1.pth",
                "final": f"receipt_counter_{model_type}_final.pth"
            }
        else:
            variant_map = {
                "best": "receipt_counter_swinv2_best.pth",
                "best_bacc": "receipt_counter_swinv2_best_bacc.pth",
                "best_f1": "receipt_counter_swinv2_best_f1.pth",
                "final": "receipt_counter_swinv2_final.pth"
            }
        model_filename = variant_map.get(args.model_variant, f"receipt_counter_{model_type}_best.pth")
        model_path = model_path / model_filename
        print(f"Using model variant: {args.model_variant} at path: {model_path}")
    
    # SwinV2 is the only supported model type
    print("Note: This evaluator only supports SwinV2 models. All models will be loaded as SwinV2.")
    
    # Use the unified evaluation function
    apply_calibration = not args.no_calibration
    
    print(f"Evaluating with model type: {model_type}")
    
    evaluate_model(
        model_path=model_path,
        test_csv=Path(args.test_csv),
        test_dir=Path(args.test_dir),
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        config_path=Path(args.config) if args.config else None,
        binary=args.binary,
        apply_calibration=apply_calibration,
        model_type=model_type
    )

if __name__ == "__main__":
    main()