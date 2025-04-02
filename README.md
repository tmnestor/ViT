# Vision Transformer (ViT) Receipt Counter

A computer vision project comparing Vision Transformer architectures (Swin-Tiny and ViT-Base) for counting receipts in images.

## Latest Updates

- **Differential Learning Rates**: Implemented separate learning rates for pretrained backbone and classifier head
- **Modern Path Handling**: Replaced `os.path` with Python's `pathlib.Path` across the codebase for more readable and robust path operations
- **Reproducibility Module**: Added `reproducibility.py` with seed control functions for consistent results across runs
- **Improved CLI**: Enhanced command-line interfaces with argument groups and shorthand options
- **Unified Training**: Consistent interfaces between Swin and ViT training scripts
- **Model Factory Pattern**: Added ModelFactory for centralized model creation, loading and saving

## Mathematical Formulation of Metrics, Weights, and Calibration

### Classification Metrics

#### Accuracy

Accuracy is the most straightforward metric, measuring the proportion of correct predictions among all 
predictions:

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = 
\frac{TP + TN}{TP + TN + FP + FN}$$

Where $TP$ = True Positives, $TN$ = True Negatives, $FP$ = False Positives, and $FN$ = False Negatives.

#### Balanced Accuracy

For imbalanced datasets, balanced accuracy provides a better performance measure by accounting for class 
imbalance:

$$\text{Balanced Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \frac{TP_i}{TP_i + FN_i}$$

Where $n$ is the number of classes, and $\frac{TP_i}{TP_i + FN_i}$ represents the recall for class $i$.

#### F1 Macro Score

The F1 score is the harmonic mean of precision and recall. F1 Macro calculates the F1 score for each class 
independently and then takes the average:

$$\text{F1 Macro} = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \times \text{Precision}_i \times 
\text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

Where:
- $\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$
- $\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$

### Class Weights and Calibration

#### Class Weights for Imbalanced Data

To address class imbalance during training, we apply inverse frequency weighting:

$$
\text{Raw Weight}_i = \frac{1}{p_i}
$$

Where $p_i$ is the prior probability (relative frequency) of class $i$ in the training data.

We then normalize these weights to maintain proper scaling:

$$\text{Normalized Weight}_i = \frac{\text{Raw Weight}_i}{\sum_{j=1}^{n} \text{Raw Weight}_j} \times n$$

The multiplication by $n$ (number of classes) ensures that the average weight remains around 1.0.

#### Bayesian Calibration Factors

Our calibration approach uses Bayesian principles to correct for overconfidence in minority classes due to 
class weighting during training (Guo et al., 2017)[^1].

For each class $i$, we compute the calibration factor as:

$$
\text{Calibration Factor}_i = p_i \times \sqrt{\frac{p_{\text{ref}}}{p_i}}
$$

Where:
- $p_i$ is the prior probability of class $i$
- $p_{\text{ref}} = \frac{1}{n}$ represents the reference probability for a balanced dataset

This formulation provides a principled Bayesian adjustment that balances the influence of the prior while 
compensating for minority classes. The square root term implements a form of Jeffreys prior[^2], which is a 
common choice in Bayesian statistics when dealing with classification problems.

For inference, we apply these calibration factors to the raw model outputs:

$$
\text{Calibrated Probability}_i = \frac{\text{Raw Probability}_i \times \text{Calibration Factor}_i \times
p_i}{\sum_{j=1}^{n} \text{Raw Probability}_j \times \text{Calibration Factor}_j \times p_j}
$$

This recalibration helps prevent the model from over-predicting minority classes, resulting in more accurate
predictions that better reflect real-world class distributions.

## System Architecture

The receipt counting system employs a flexible configuration approach that dynamically sets the number of classes based on configuration parameters rather than hardcoded values. This design allows the system to adapt to changing class probability distributions without requiring code modifications.

### Key Components:

1. **Configuration System**
   - Dynamically loads class distribution from JSON config files
   - Automatically derives calibration factors from class distribution
   - Provides silent loading option to prevent repetitive logging messages

2. **Model Creation**
   - ViT and Swin Transformer models instantiated with dynamic class counts
   - Configuration-based weight and head dimension determination
   - Customized classification heads with improved regularization

3. **Training Pipeline**
   - Dynamic validation metrics calculation based on class count
   - Adaptive plot generation for confusion matrices and class distributions
   - Early stopping based on balanced accuracy and F1 macro improvements

## Vision Transformer (ViT) Receipt Counter

A computer vision project comparing Vision Transformer architectures (Swin-Tiny and ViT-Base) for counting receipts in images.

## Project Overview

This project compares the Swin-Tiny and ViT-Base vision transformer architectures to determine which can more accurately count the number of receipts in an image. Both models use a classification approach to predict the number of receipts (0-5) in synthetic collage images.

## Setup

### Option 1: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create conda environment
conda env create -n vit_env python=3.11
conda activate vit_env

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- **Core Model Files:**
  - `model_factory.py` - Factory pattern for creating and loading models
  - `transformer_swin.py` - Swin-Tiny transformer implementation (legacy)
  - `transformer_vit.py` - ViT-Base transformer implementation (legacy)
  - `receipt_processor.py` - Image preprocessing utilities
  - `config.py` - Centralized configuration system

- **Unified Dataset & Training Modules:**
  - `datasets.py` - Unified dataset classes for ViT and Swin models
  - `training_utils.py` - Shared training, validation, and evaluation utilities

- **Data Generation & Preparation:**
  - `create_receipt_collages.py` - Generate synthetic receipt collages on table surfaces
  - `prepare_collage_dataset.py` - Prepare datasets from collage images
  - `download_test_images.py` - Utility to download/process SRD receipt dataset

- **Training & Evaluation:**
  - `train_swin_classification.py` - Train the Swin-Tiny receipt counter
  - `train_vit_classification.py` - Train the ViT-Base receipt counter
  - `evaluate_swin_counter.py` - Evaluate Swin-Tiny model performance
  - `evaluate_vit_counter.py` - Evaluate ViT-Base model performance
  - `individual_image_tester.py` - Process individual images through trained models

- **Additional Utilities:**
  - `simple_receipt_counter.py` - Simplified model for quick testing
  - `batch_processor.py` - Batch processing utilities
  - `test_images_demo.py` - Testing with multiple sample images

## Workflow

### 1. Generate Synthetic Training Data

You can generate realistic collages of receipts on various table surfaces using the collage generator:

```bash
# Generate 300 collages with 0-5 receipts based on default configuration
python create_receipt_collages.py --num_collages 300

# Generate with specific probability distribution for receipt counts
# Format: p0,p1,p2,p3,p4,p5 where pN is probability of having N receipts
python create_receipt_collages.py --num_collages 300 --count_probs 0.3,0.2,0.2,0.1,0.1,0.1

# Use distribution from a configuration file
python create_receipt_collages.py --num_collages 300 --config custom_config.json

# Full control over output
python create_receipt_collages.py --num_collages 200 \
  --canvas_width 1600 --canvas_height 1200 \
  --count_probs 0.3,0.2,0.2,0.1,0.1,0.1
```

The collages will be saved in a `receipt_collages` directory with filenames indicating how many receipts are in each image. The current version places receipts on various background surfaces with shadows and realistic color blending. Receipts have limited rotation (±15 degrees) and may have slight overlapping to create a more challenging dataset.

### 2. Prepare the Dataset

```bash
# Prepare dataset from synthetic collages
python prepare_collage_dataset.py --collage_dir receipt_collages --output_dir receipt_dataset
```

This splits the data into training and validation sets, and creates CSV files with image paths and receipt counts.

### 3. Download Real Test Images (Optional)

```bash
python download_test_images.py
```

The SRD dataset requires manual download due to access restrictions. The script provides instructions on how to obtain the dataset and will process it once downloaded.

### 4. Train the Models

```bash
# Train Swin-Tiny model with default configuration
python train_swin_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                           --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                           --epochs 20 --batch_size 32 --output_dir models \
                           --lr 5e-4 --backbone_lr_multiplier 0.02

# Train with custom class distribution
python train_swin_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                           --class_dist "0.4,0.3,0.15,0.05,0.05,0.05" \
                           --epochs 20 --batch_size 32 --output_dir models

# Train with configuration file
python train_swin_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                           --config custom_config.json \
                           --epochs 10 --batch_size 32 --output_dir models
                           
# Train ViT-Base model with default configuration
python train_vit_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                          --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                          --epochs 20 --batch_size 32 --output_dir models \
                          --lr 5e-5

# Train with custom differential learning rates
python train_vit_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                          --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                          --epochs 30 --batch_size 16 --output_dir models \
                          --lr 1e-4 --backbone_lr_multiplier 0.05

# Train with gradient clipping and differential learning rates
python train_swin_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                           --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                           --epochs 20 --batch_size 16 --output_dir models \
                           --lr 5e-5 --backbone_lr_multiplier 0.1 --grad_clip 2.0

```

Training generates evaluation metrics, confusion matrices, accuracy plots, and saves models to the `models` directory for both architectures. The classification approach uses a ReduceLROnPlateau scheduler to prevent erratic training behavior.

#### Hugging Face Transformers Implementation Details

The training scripts utilize Hugging Face transformers with several optimization techniques:

```python
# Model initialization through our factory pattern
from model_factory import ModelFactory

# Create model with the right type (swin or vit)
model = ModelFactory.create_transformer(model_type="swin", pretrained=True).to(device)

# Training loop with differential learning rates and gradient clipping
for images, targets in dataloader:
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass with Hugging Face model
    outputs = model(images)
    logits = outputs.logits  # Hugging Face models return an object with logits
    
    # Calculate loss with class weighting
    loss = criterion(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
    
    # Update with AdamW optimizer (uses different learning rates for backbone vs classifier)
    optimizer.step()
```

### 5. Evaluate the Models

```bash
# Evaluate Swin-Tiny model trained with Hugging Face implementation
python evaluate_swin_counter.py --model models/receipt_counter_swin_best.pth \
                                 --test_csv receipt_dataset/val.csv \
                                 --test_dir receipt_dataset/val \
                                 --output_dir evaluation/swin_tiny


# Evaluate ViT-Base model trained with Hugging Face implementation
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth \
                             --test_csv receipt_dataset/val.csv \
                             --test_dir receipt_dataset/val \
                             --output_dir evaluation/vit_base

```

The evaluation scripts generate detailed metrics (including per-class accuracy, confusion matrices, balanced accuracy, and F1 scores) and visualizations in their respective output directories, allowing for direct comparison between the model architectures and implementations.

#### Balanced Accuracy vs F1-Macro Score

For this multi-class classification task with potential class imbalance, two key metrics are used:

- **Balanced Accuracy**: The average of recall obtained on each class. It's calculated as the average of the true positive rate for each class, making it robust to class imbalance. Balanced accuracy treats all classes equally regardless of their support (number of samples).
  
- **F1-Macro Score**: The unweighted average of the F1 scores for each class. F1 score is the harmonic mean of precision and recall, calculated as 2 * (precision * recall) / (precision + recall). F1-macro treats all classes equally regardless of their support.

**Comparison:**
- Balanced accuracy focuses only on recall (true positive rate), measuring how well each class is detected
- F1-macro considers both precision and recall, balancing between detecting all positives and avoiding false positives
- When false positives are costly, F1-macro may be preferred
- When false negatives are costly, balanced accuracy might be more informative
- Both metrics are robust to class imbalance, unlike overall accuracy which can be misleading when classes are imbalanced

For the receipt counting task, both metrics provide valuable insights. Balanced accuracy helps understand if the model can correctly identify images with all possible receipt counts (0-5), while F1-macro additionally considers how precisely the model makes these predictions.


### 6. Test on Individual Images

```bash
# Test Swin-Tiny model on a single image with default configuration
python individual_image_tester.py --image receipt_collages/collage_254_3_receipts.jpg --model models/receipt_counter_swin_best.pth 

# Test ViT-Base model with a custom configuration file
python individual_image_tester.py --image test_images/collage_254_3_receipts.jpg--model models/receipt_counter_vit_best.pth --config custom_config.json

# Test with different model variants (best balanced accuracy, best F1)
python individual_image_tester.py --image test_images/collage_254_3_receipts.jpg --model models/receipt_counter_vit_best_f1.pth
```

## Hardware Acceleration

This project can utilize hardware acceleration:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

The code automatically detects and uses the best available device through the `device_utils` module. For Apple Silicon (M1/M2/M3) Mac users, the MPS backend is used with fallback to CPU for operations not supported by MPS.

### Using the Device Utilities

The project provides device abstraction through the `device_utils.py` module:

```python
from device_utils import get_device, move_to_device, get_device_info

# Get the best available device
device = get_device()

# Move model and data to device in one step
model, data = move_to_device(model, data)

# Get detailed information about the current device
device_info = get_device_info()
print(f"Using {device_info['device_type']} with PyTorch {device_info['pytorch_version']}")
```

This abstraction ensures consistent device handling across the entire codebase and optimizes hardware usage.

### Modern Path Handling

The project uses `pathlib.Path` instead of `os.path` for modern, object-oriented path handling:

```python
from pathlib import Path

# Create a path object
model_dir = Path("models")

# Create directories with parents
output_path = Path("output/results")
output_path.mkdir(parents=True, exist_ok=True)

# Join paths with / operator
model_path = model_dir / "receipt_counter_swin_best.pth"

# Path properties and methods
if model_path.exists() and model_path.is_file():
    print(f"Found model: {model_path.name}")
    print(f"In directory: {model_path.parent}")
    print(f"File extension: {model_path.suffix}")

# Globbing files
image_dir = Path("receipt_collages")
for image_path in image_dir.glob("*.jpg"):
    print(f"Processing {image_path.stem}")

# Handling paths consistently on all platforms
csv_path = Path("data") / "train.csv"
```

The `pathlib` integration improves code readability, portability across operating systems, and reduces errors from string-based path manipulation.

### Reproducibility

The project ensures reproducible results through the `reproducibility.py` module:

```python
from reproducibility import set_seed, get_reproducibility_info, is_deterministic

# Set random seed from configuration
seed_info = set_seed()
print(f"Using seed {seed_info['seed']}, deterministic: {seed_info['deterministic']}")

# Get current reproducibility settings
repro_info = get_reproducibility_info()
print(f"PYTHONHASHSEED: {repro_info['python_hashseed']}")

# Check if using deterministic mode
if is_deterministic():
    print("Running in deterministic mode (slower but reproducible)")
else:
    print("Running in performance mode (faster but not fully reproducible)")
```

To ensure completely reproducible results:

1. Set a fixed random seed via configuration or command line parameter
2. Enable deterministic mode
3. Use a single worker for data loading when using DataLoader

Example usage with command line parameters:

```bash
# Train with fixed seed, deterministic algorithms, and differential learning rates
python train_vit_classification.py --seed 42 --deterministic --backbone_lr_multiplier 0.05
```

## Model Architectures

The project compares two vision transformer architectures using Hugging Face Transformers implementations:

### Swin-Tiny Transformer
- **Architecture**: Hierarchical vision transformer using shifted windows for efficient processing
- **Backbone**: Swin-Tiny pre-trained on ImageNet
- **Parameters**: ~28 million 
- **Input Size**: 224x224 pixels
- **Output**: 6 classes (0-5 receipts)
- **Loss Function**: Cross Entropy Loss with label smoothing
- **Optimization**: AdamW with ReduceLROnPlateau scheduler and gradient clipping
- **Implementation**: Hugging Face `microsoft/swin-tiny-patch4-window7-224` with custom classifier


### ViT-Base Transformer
- **Architecture**: Standard Vision Transformer with non-overlapping patches 
- **Backbone**: ViT-Base-16 pre-trained on ImageNet
- **Parameters**: ~86 million
- **Input Size**: 224x224 pixels
- **Output**: 6 classes (0-5 receipts)
- **Loss Function**: Cross Entropy Loss with label smoothing
- **Optimization**: AdamW with ReduceLROnPlateau scheduler and gradient clipping
- **Implementation**: Hugging Face `google/vit-base-patch16-224` with custom classifier


Both models are fine-tuned using the AdamW optimizer with differential learning rates:
- **Classifier Head**: Higher learning rate (default: 5e-5) for the randomly initialized classification layers
- **Backbone**: Lower learning rate (default: 0.1× classifier rate) for the pretrained transformer backbone

This differential learning rate approach is a transfer learning best practice that:
1. Preserves the valuable pretrained features in the backbone with a gentler learning rate
2. Allows the new classification head to learn quickly with a higher learning rate
3. Can be tuned via the `--backbone_lr_multiplier` parameter to optimize performance

Additional optimization techniques include increased weight decay (0.05), BatchNorm and Dropout regularization, and a ReduceLROnPlateau scheduler (patience=2, factor=0.5) to stabilize training. Gradient clipping is applied to prevent exploding gradients.

## Configuration System

This project includes a comprehensive centralized configuration system to make the receipt counter adaptable to changes in class distribution and model architecture parameters. This is especially important in production environments where the real-world distribution of classes may differ from the training distribution, or when you need to modify hyperparameters for different deployment scenarios.

### Single Source of Truth

All default configuration values are defined in `config.py` as constants, making it the single source of truth for the project. This approach eliminates inconsistencies between code and external configuration files.

```python
# Default class distribution and parameters from config.py
DEFAULT_CLASS_DISTRIBUTION = [0.4, 0.2, 0.2, 0.1, 0.1]
DEFAULT_BINARY_DISTRIBUTION = [0.6, 0.4]

DEFAULT_MODEL_PARAMS = {
    # Image parameters
    "image_size": 224,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    # Classifier architecture
    "classifier_dims": [768, 512, 256],
    "dropout_rates": [0.4, 0.4, 0.3],
    # Training parameters
    "batch_size": 16,
    "learning_rate": 5e-5,
    "backbone_lr_multiplier": 0.1,
    "weight_decay": 0.01,
    "num_workers": 4,
    "label_smoothing": 0.1,
    # And more...
}
```

You can save the current configuration to a JSON file for reference or sharing, but the system does not rely on external files for initialization.

### Environment Variables

To override defaults without modifying the code, you can use environment variables:

```bash
# Set class distribution
export RECEIPT_CLASS_DIST="0.4,0.2,0.2,0.1,0.1"

# Set model parameters directly
export RECEIPT_IMAGE_SIZE="256"
export RECEIPT_BATCH_SIZE="32" 
export RECEIPT_LEARNING_RATE="1e-5"
export RECEIPT_BACKBONE_LR_MULTIPLIER="0.05"
export RECEIPT_NUM_WORKERS="8"
export RECEIPT_WEIGHT_DECAY="0.005"
export RECEIPT_LABEL_SMOOTHING="0.05"
export RECEIPT_GRADIENT_CLIP="2.0"
```

The calibration factors will be automatically derived from the class distribution.

### Using the Configuration System

All core scripts support direct class distribution overrides via command line arguments:

```bash
# Direct override of class distribution
python train_vit_classification.py --class_dist "0.25,0.25,0.2,0.1,0.1,0.1"

# Set backbone learning rate multiplier (controls differential learning rates)
python train_swin_classification.py --backbone_lr_multiplier 0.05

# Set binary mode
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth --binary
```

For more complex configuration, use environment variables:

```bash
# Set image size, batch size and learning rates for training
export RECEIPT_IMAGE_SIZE=256
export RECEIPT_BATCH_SIZE=32
export RECEIPT_LEARNING_RATE=1e-4
export RECEIPT_BACKBONE_LR_MULTIPLIER=0.05
export RECEIPT_NUM_WORKERS=8
python train_vit_classification.py

# Use environment variables for class distribution
export RECEIPT_CLASS_DIST="0.3,0.2,0.2,0.1,0.1"
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth
```

You can export the current configuration to a JSON file for reference:

```python
from config import get_config
config = get_config()
config.save_to_file("current_config.json")
```

### How the Configuration System Works

The configuration system is implemented as a singleton in `config.py`:

1. When imported, it initializes with default values directly from constants in the file
2. It checks for environment variables that might override these defaults
3. Based on the class distribution, it automatically calculates:
   - Inverse weights for loss function
   - Normalized weights
   - Scaled weights for CrossEntropyLoss
   - **Bayesian calibration factors** for inference-time calibration
4. It provides access to model parameters with safe defaults:
   - Model architecture parameters (image size, classifier dimensions)
   - Training parameters (batch size, learning rate, workers)
   - Optimizer settings (weight decay, gradient clipping)
   - Scheduler parameters (factor, patience, minimum learning rate)
   - Early stopping settings

When exporting to JSON, the configuration looks like this:

```json
{
  "binary_mode": false,
  "class_distribution": [0.4, 0.2, 0.2, 0.1, 0.1],
  "model_params": {
    "image_size": 224,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "backbone_lr_multiplier": 0.1,
    "num_workers": 4,
    "label_smoothing": 0.1,
    "classifier_dims": [768, 512, 256],
    "dropout_rates": [0.4, 0.4, 0.3],
    "weight_decay": 0.01,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 2,
    "min_lr": 1e-6,
    "gradient_clip_value": 1.0,
    "early_stopping_patience": 8
  },
  "derived_calibration_factors": [1.0, 0.894, 0.894, 0.632, 0.632]
}
```

During training and inference, the system provides tensors and parameters ready for use with PyTorch:

```python
from config import get_config

# Get configuration singleton
config = get_config()

# Get class weights for loss function
weights = config.get_class_weights_tensor(device)
label_smoothing = config.get_model_param("label_smoothing", 0.1)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

# Get automatically derived calibration factors for inference
calibration = config.get_calibration_tensor(device)
class_prior = config.get_class_prior_tensor(device)

# Get model architecture parameters
image_size = config.get_model_param("image_size", 224)
classifier_dims = config.get_model_param("classifier_dims", [768, 512, 256])
dropout_rates = config.get_model_param("dropout_rates", [0.4, 0.4, 0.3])

# Get differential learning rate settings
learning_rate = config.get_model_param("learning_rate", 5e-5)
backbone_lr_multiplier = config.get_model_param("backbone_lr_multiplier", 0.1)
backbone_lr = learning_rate * backbone_lr_multiplier
print(f"Using learning rates - Classifier: {learning_rate}, Backbone: {backbone_lr}")

# To understand the derivation of calibration factors
explanation = config.explain_calibration()
print(explanation)
```

The `explain_calibration()` method provides a detailed breakdown of how each calibration factor is derived from the class distribution, making the system transparent and interpretable. All hardcoded values like image sizes, learning rates, batch sizes, and architecture parameters are now configurable through this centralized system.

## Class Weighting and Calibration

This project implements a comprehensive class weighting system to address the imbalanced distribution of receipt counts in real-world scenarios. The implementation uses both training-time weighting and inference-time calibration.

### Class Imbalance Problem

In real-world applications, images with 0-1 receipts are much more common than images with 2-5 receipts. Our dataset reflects this imbalance with an approximate distribution of:

```
Class 0 (0 receipts): 30% of samples
Class 1 (1 receipt):  20% of samples
Class 2 (2 receipts): 20% of samples
Class 3 (3 receipts): 10% of samples
Class 4 (4 receipts): 10% of samples
Class 5 (5 receipts): 10% of samples
```

This imbalance causes models to perform well on the majority classes (0-1) but poorly on minority classes (2-5).

### Training-Time Class Weighting

During training, we apply inverse weighting to give more importance to underrepresented classes:

```python
# Get class weights from configuration
config = get_config()
normalized_weights = config.get_class_weights_tensor(device)

# Apply to loss function
criterion = nn.CrossEntropyLoss(weight=normalized_weights, label_smoothing=0.1)
```

This weighting scheme forces the model to pay more attention to classes 2-5 during training, resulting in better balanced accuracy and F1-macro scores.

### Label Smoothing

We also apply label smoothing (0.1) to create soft targets rather than hard one-hot encoded targets. This:
- Prevents the model from becoming overly confident
- Improves generalization
- Adds regularization, especially helpful for minority classes

### Bayesian Inference-Time Calibration

While the class weighting strategy improves performance on underrepresented classes during training, it can cause the model to over-predict higher count classes during inference. To address this, we implement a principled Bayesian calibration system in the inference pipeline:

```python
# Get calibration parameters from configuration
config = get_config()
class_prior = config.get_class_prior_tensor(device)
calibration_factors = config.get_calibration_tensor(device)

# Apply calibration
calibrated_probs = raw_probs * calibration_factors * class_prior
calibrated_probs = calibrated_probs / calibrated_probs.sum()  # Re-normalize
```

#### Bayesian Derivation of Calibration Factors

Our calibration factors are derived from the class distribution using a principled Bayesian approach. The formula for deriving calibration factors is:

```
calibration_factor = prior_probability * sqrt(reference_probability / prior_probability)
```

Where:
- `prior_probability` is the class frequency in our dataset
- `reference_probability` is the balanced distribution (1/num_classes)
- `sqrt(reference_probability / prior_probability)` is the adjustment term

This formula has a solid mathematical foundation in Bayesian statistics:

1. **Bayesian Adjustment**: The square root term moderates the influence of the prior, creating a balanced weighting between the raw model output and the true class distribution

2. **Maximum Entropy Principle**: The formula maximizes the information gain while respecting the original distribution constraints

3. **Temperature Scaling**: The square root effectively applies a "temperature" to the calibration, preventing over-correction

After computing these raw factors, we normalize them by dividing by the maximum value, giving a cleaner range of values between 0 and 1 while preserving the relative relationships.

Our two-stage approach (weight during training, calibrate during inference) results in a model that:
1. Learns effectively from minority classes
2. Produces calibrated predictions that match the true class distribution
3. Maintains high balanced accuracy while avoiding over-prediction of rare classes

The calibration factors are automatically derived from the class distribution in the configuration system, ensuring they always stay mathematically consistent with the data distribution.

### Implementation Details

- **Training Scripts**: Both `train_vit_classification.py` and `train_swin_classification.py` implement class weighting
- **Evaluation Scripts**: Both evaluation scripts are aware of the class weights but evaluate using unweighted metrics
- **Inference Scripts**: `individual_image_tester.py` applies calibration to the model outputs

### Considerations for Deployment

When deploying this model in a production environment:

1. **Class Distribution Changes**: If the real-world class distribution changes, you only need to update the class distribution in the configuration file. The system will automatically derive appropriate calibration factors using the Bayesian formula.

2. **Monitoring Distribution Drift**: Monitor real-world class distributions over time. If they shift significantly, update the distribution in the configuration to keep the model well-calibrated.

3. **Explainable Calibration**: Use `config.explain_calibration()` to understand how the calibration factors are derived from your current class distribution. This transparency helps in auditing and explaining model behavior.

4. **Custom Thresholds**: For applications with specific requirements (high precision vs. high recall), additional thresholding can be applied to the calibrated probabilities.

5. **Validation**: While our Bayesian formula provides theoretically sound calibration, it's always good practice to validate its effectiveness on your specific dataset.


## Credits

This project uses the following vision transformer architectures:

- **Swin Transformer**: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

- **Vision Transformer (ViT)**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR).

The project uses implementations from:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Provides state-of-the-art machine learning models for various tasks

## Key Module Usage

The code has been refactored into several reusable modules with consistent interfaces:

### Model Factory Module

The code uses a unified model factory that handles both ViT and Swin model creation:

```python
# Import the model factory
from model_factory import ModelFactory

# Create models using the factory
vit_model = ModelFactory.create_transformer(model_type="vit", pretrained=True)
swin_model = ModelFactory.create_transformer(model_type="swin", pretrained=True)

# Save models
ModelFactory.save_model(vit_model, "path/to/vit_model.pth")
ModelFactory.save_model(swin_model, "path/to/swin_model.pth")

# Load models
loaded_vit = ModelFactory.load_model("path/to/vit_model.pth", model_type="vit")
loaded_swin = ModelFactory.load_model("path/to/swin_model.pth", model_type="swin")
```

Legacy interfaces are still supported for backward compatibility:

```python
# Legacy transformer interfaces
from model_factory import create_vit_transformer, create_swin_transformer
from model_factory import load_vit_model, load_swin_model, save_model
```

### Unified Dataset Module

A unified dataset module provides consistent data loading for both ViT and Swin models:

```python
# Import dataset classes
from datasets import ReceiptDataset, create_data_loaders

# Create data loaders with a single function call
train_loader, val_loader, num_train, num_val = create_data_loaders(
    train_csv="receipt_dataset/train.csv",
    train_dir="receipt_dataset/train",
    val_csv="receipt_dataset/val.csv",
    val_dir="receipt_dataset/val",
    batch_size=32,
    augment_train=True,
    binary=False
)
```

The dataset module handles:
- Consistent data loading for both model types
- Train/validation split when no separate validation set is provided
- Image augmentation during training
- Binary or multiclass labels based on configuration
- Automatic handling of missing files with fallback paths

### Training Utilities Module

A training utilities module provides shared functionality for model training, validation, and evaluation:

```python
# Import training utilities
from training_utils import (
    validate, print_validation_results, plot_confusion_matrix, 
    plot_training_curves, plot_evaluation_metrics,
    ModelCheckpoint, EarlyStopping
)

# Validate model and get metrics dictionary
metrics = validate(model, dataloader, criterion, device)

# Use model checkpoint with multiple tracked metrics
checkpoint = ModelCheckpoint(
    output_dir="models",
    metrics=["balanced_accuracy", "f1_macro"],
    mode="max",
    verbose=True
)
checkpoint.check_improvement(metrics, model, model_type="vit")

# Use early stopping with configurable patience
early_stopping = EarlyStopping(patience=8, mode="max")
if early_stopping.check_improvement(metrics["balanced_accuracy"]):
    print("Early stopping triggered")

# Generate evaluation plots
plot_evaluation_metrics(metrics, output_dir="evaluation")
```

The training utilities module standardizes:
- Model validation across different model types
- Performance metrics calculation
- Model checkpointing based on multiple metrics
- Early stopping based on various criteria
- Visualization of results and training progress
- Consistent output formats for reuse

## References

[^1]: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1321-1330).

[^2]: Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 186(1007), 453-461.