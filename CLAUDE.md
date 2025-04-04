# Vision Transformer Receipt Counter Project

## Key Commands

### Training Commands

```bash
# Train ViT-Base model with reproducibility
python train_vit_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                          -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                          -e 20 -b 8 -o models -s 42 -d \
                          -l 5e-5 

# Train Swin-Tiny model with reproducibility
python train_swin_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                           -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                           -e 20 -b 8 -o models -s 42 -d \
                           -l 5e-5

# Resume training from a checkpoint
python train_vit_classification.py -r models/receipt_counter_vit_best.pth \
                          -e 10 -b 8 -o models

# Dry run to validate configuration
python train_swin_classification.py --dry-run -e 30 -b 8 -s 42
```

### CLI Options

The training scripts now have an improved CLI with argument groups:

#### Data Options
- `-tc/--train_csv`: Path to training CSV file
- `-td/--train_dir`: Directory containing training images
- `-vc/--val_csv`: Path to validation CSV file
- `-vd/--val_dir`: Directory containing validation images
- `--no-augment`: Disable data augmentation during training

#### Training Options
- `-e/--epochs`: Number of training epochs
- `-b/--batch_size`: Batch size for training
- `-l/--lr`: Learning rate for classifier head
- `-blrm/--backbone_lr_multiplier`: Multiplier for backbone learning rate (default: 0.1)
- `-gc/--grad_clip`: Gradient clipping max norm (default: 1.0)
- `-wd/--weight_decay`: Weight decay for optimizer
- `-ls/--label_smoothing`: Label smoothing factor
- `-o/--output_dir`: Directory to save trained model
- `-c/--config`: Path to configuration JSON file
- `-r/--resume`: Resume training from checkpoint file
- `-bin/--binary`: Train as binary classification
- `--dry-run`: Validate configuration without training
- `--class_dist`: Comma-separated class distribution

#### Reproducibility Options
- `-s/--seed`: Random seed for reproducibility
- `-d/--deterministic`: Enable deterministic mode

### Evaluation Commands

```bash
# Evaluate ViT model
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth \
                             --test_csv receipt_dataset/val.csv \
                             --test_dir receipt_dataset/val \
                             --output_dir evaluation/vit_base

# Evaluate SwinV2 model
python evaluate_swinv2_classifier.py --model models/receipt_counter_swinv2_best.pth \
                                 --test_csv receipt_dataset/val.csv \
                                 --test_dir receipt_dataset/val \
                                 --output_dir evaluation/swinv2_tiny
```

### Testing on Individual Images

```bash
# Test a single image with ViT model
python individual_image_tester.py --image receipt_collages/collage_014_2_receipts.jpg --model models/receipt_counter_vit_best.pth

# Test a single image with Swin model
python individual_image_tester.py --image receipt_collages/collage_014_2_receipts.jpg --model models/receipt_counter_swin_best.pth
```

### Testing on Multiple Images

```bash
# Show sample images
python test_images_demo.py --image_dir receipt_collages --samples 4 --mode show

# Process sample images with Swin model
python test_images_demo.py --image_dir receipt_collages --samples 4 --mode process --model models/receipt_counter_swin_best.pth
```

## Project Structure

### Core Modules:
- `model_factory.py` - Factory pattern for creating and loading models
- `datasets.py` - Unified dataset implementation 
- `training_utils.py` - Shared training, validation, and evaluation utilities
- `config.py` - Centralized configuration system
- `device_utils.py` - Device abstraction for hardware acceleration 
- `evaluation.py` - Unified evaluation functionality
- `reproducibility.py` - Seed control and deterministic behavior

### Training Scripts:
- `train_vit_classification.py` - Train the ViT-Base model
- `train_swin_classification.py` - Train the Swin-Tiny model

### Evaluation Scripts:
- `evaluate_vit_counter.py` - Evaluate the ViT-Base model
- `evaluate_swin_counter.py` - Evaluate the Swin-Tiny model

### Testing Scripts:
- `individual_image_tester.py` - Test a single image
- `test_images_demo.py` - Test multiple images

### Data Generation:
- `create_receipt_collages.py` - Generate synthetic receipt collages
- `create_collage_dataset.py` - Create datasets from collage images

## Recent Refactoring

The codebase has been refactored to:
1. Implement a model factory pattern in `model_factory.py`
2. Unify dataset handling in `datasets.py` with a single `ReceiptDataset` class
3. Extract training utilities into `training_utils.py`
4. Standardize validation, checkpointing, and early stopping logic 
5. Implement consistent dictionary-based metrics return values
6. Use `pathlib.Path` for modern path handling instead of `os.path`
7. Add device abstraction with `device_utils.py` for consistent hardware acceleration
8. Implement reproducibility with `reproducibility.py` and `set_seed()` function
9. Improve CLI interfaces with argument groups and shorthand options
10. Add support for checkpoint resuming via the `--resume` parameter

## Environment Variables

You can use these environment variables to override default configurations:

```bash
# Class distribution
export RECEIPT_CLASS_DIST="0.4,0.2,0.2,0.1,0.1"

# Model parameters
export RECEIPT_IMAGE_SIZE="256"
export RECEIPT_BATCH_SIZE="8" 
export RECEIPT_LEARNING_RATE="1e-5"
export RECEIPT_NUM_WORKERS="8"
export RECEIPT_WEIGHT_DECAY="0.005"
export RECEIPT_LABEL_SMOOTHING="0.05"

# Reproducibility settings
export RECEIPT_RANDOM_SEED="42"
export RECEIPT_DETERMINISTIC_MODE="true"
```

## Reproducibility

The project now includes a centralized reproducibility module for consistent results:

- Set random seeds for all libraries (Python, NumPy, PyTorch)
- Optional deterministic mode for complete reproducibility  
- Command-line options via `--seed` and `--deterministic` flags
- Environment variable configuration via `RECEIPT_RANDOM_SEED` and `RECEIPT_DETERMINISTIC_MODE`
- Default seed is 42 if not specified

Note: Full deterministic mode may impact performance, especially on GPUs.

## Environment Setup

The project uses a Conda environment specified in `environment.yml`. To set up the environment:

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate vit_env
```

Key dependencies:
- Python 3.11
- PyTorch 2.6.0
- torchvision 0.21.0
- transformers 4.49.0
- pandas 2.2.3
- scikit-learn 1.6.1
- albumentations 2.0.5
- matplotlib 3.10.1