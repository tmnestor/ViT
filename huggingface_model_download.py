"""
Utility script to pre-download model weights and configuration from HuggingFace.
This allows any supported model to be used in offline mode.
"""

import argparse
import os
from pathlib import Path

from transformers import AutoConfig, AutoImageProcessor, AutoModel


def download_model(model_name, output_dir=None):
    """
    Download model weights, configuration, and processor for offline use.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save a copy of the cached files (optional)
    """
    print(f"Downloading {model_name}...")

    # Download the model config
    print("Downloading model configuration...")
    config = AutoConfig.from_pretrained(model_name)
    print("Model configuration downloaded successfully")

    # Download the model weights
    print("Downloading model weights...")
    model = AutoModel.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    print("Model weights downloaded successfully")

    # Download the image processor
    print("Downloading image processor...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        print("Image processor downloaded successfully")
    except Exception as e:
        print(f"Warning: Could not download image processor: {e}")
        print("This may be normal for some model types")
        processor = None

    # If output directory is specified, save the model files there as well
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {output_path}...")
        model.save_pretrained(output_path)
        if processor:
            processor.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    # Print cache location
    print(
        "\nModel is now cached locally. You can use --offline mode with the training script."
    )

    # Get the cache directory path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    print(f"Default cache location: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model for offline use"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name to download"
    )
    parser.add_argument(
        "--output_dir", 
        help="Optional directory to save model files"
    )

    args = parser.parse_args()
    
    download_model(args.model_name, args.output_dir)

    # Print instructions for offline use
    print("\nTo use the downloaded model in offline mode:")
    
    # Instructions for using the model in offline mode with specified directory
    output_path_arg = ""
    if args.output_dir:
        output_path_arg = f"--pretrained_model_dir {args.output_dir}"
    
    print(
        f"python train_vit_classification.py --model_name {args.model_name} --offline {output_path_arg} \\\n"
        "                        --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \\\n"
        "                        --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \\\n"
        "                        --output_dir /path/to/trained/models --epochs 20"
    )


if __name__ == "__main__":
    main()