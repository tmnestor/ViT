"""
Utility script to pre-download model weights and configuration from HuggingFace.
This allows any supported model to be used in offline mode.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Set environment variable to disable NumPy 2.x compatibility warnings
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# Check if running in conda environment
if 'CONDA_PREFIX' not in os.environ:
    print("WARNING: Not running in a conda environment.")
    print("Please activate the vit_env conda environment first:")
    print("conda activate vit_env")
    sys.exit(1)

# Create a simple script to download the model directly using huggingface_hub
try:
    from huggingface_hub import snapshot_download
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"Error installing huggingface_hub: {e}")
        sys.exit(1)


def download_model(model_name, output_dir=None):
    """
    Download model weights, configuration, and processor for offline use.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save a copy of the cached files (optional)
    """
    print(f"Downloading {model_name}...")

    try:
        # Get the local directory where the model will be cached
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        
        # Download the model files directly using huggingface_hub
        print(f"Downloading model files for {model_name}...")
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            revision="main",
            ignore_patterns=["*.msgpack", "*.safetensors.index.json"],
            resume_download=True
        )
        
        print(f"Model downloaded successfully to: {local_dir}")
        
        # If output_dir was not specified but we still want to know where the files are
        if not output_dir:
            # Model is in huggingface cache
            model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
            if os.path.exists(model_path):
                print(f"Model is also cached at: {model_path}")
                
    except Exception as e:
        print(f"Error downloading model: {e}")
        if "NumPy" in str(e) or "numpy" in str(e).lower():
            print("This appears to be a NumPy 2.x compatibility issue.")
            print("Solutions:")
            print("1. Use a conda environment with numpy<2.0")
            print("2. Run: conda install -y numpy=1.24.3")
        elif "torch" in str(e).lower() or "torchvision" in str(e).lower():
            print("This appears to be a PyTorch/TorchVision compatibility issue.")
            print("Solutions:")
            print("1. Create a new conda environment with compatible versions:")
            print("   conda create -n model_env python=3.10 torch torchvision transformers -c pytorch -c huggingface -c conda-forge")
        sys.exit(1)

    # Print offline usage instructions
    print(
        "\nModel is now cached locally. You can use --offline mode with the training script."
    )
    
    # Print the directory where the model is saved
    if output_dir:
        print(f"Model saved to: {output_dir}")
    else:
        print(f"Default cache location: {os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')}")


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