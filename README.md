# Vision Transformer (ViT) Receipt Counter

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
  - `transformer_swin.py` - Swin-Tiny transformer model implementation
  - `transformer_vit.py` - ViT-Base transformer model implementation
  - `receipt_processor.py` - Image preprocessing utilities

- **Data Generation & Preparation:**
  - `create_receipt_collages.py` - Generate synthetic receipt collages on table surfaces
  - `prepare_collage_dataset.py` - Prepare datasets from collage images
  - `download_test_images.py` - Utility to download/process SRD receipt dataset

- **Training & Evaluation:**
  - `train_swin_classification.py` - Train the Swin-Tiny receipt counter using classification (0-5 classes)
  - `train_vit_classification.py` - Train the ViT-Base receipt counter using classification (0-5 classes)
  - `evaluate_swin_counter.py` - Evaluate Swin-Tiny model performance with metrics
  - `evaluate_vit_counter.py` - Evaluate ViT-Base model performance with metrics
  - `demo.py` - Process individual images through the trained model

- **Additional Utilities:**
  - `torchvision_demo.py` - Demo of pre-trained ViT from torchvision
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
                           --lr 5e-5

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

```

Training generates evaluation metrics, confusion matrices, accuracy plots, and saves models to the `models` directory for both architectures. The classification approach uses a ReduceLROnPlateau scheduler to prevent erratic training behavior.

#### Implementation Guide

The training scripts should be modified to use Hugging Face implementations as the default:

```python
# Add to argument parser
parser.add_argument("--use_torchvision", action="store_true", help="Use torchvision implementation instead of Hugging Face")
parser.add_argument("--grad_clip", type=float, default=None, help="Apply gradient clipping with specified max norm")

# Then in the model initialization section:
if args.use_torchvision:
    model = ReceiptCounter(pretrained=True, num_classes=6).to(device)
    print("Initialized torchvision Swin-Tiny model for receipt counting")
else:
    from receipt_counter import create_hf_receipt_counter
    model = create_hf_receipt_counter(pretrained=True, num_classes=6).to(device)
    print("Initialized Hugging Face Swin-Tiny model for receipt counting")

# Later in training loop, add gradient clipping if requested:
for batch in dataloader:
    # ... forward pass, loss calculation, etc. ...
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping if specified
    if args.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
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
python individual_image_tester.py --image receipt_collages/collage_280_2_receipts.jpg --model models/receipt_counter_swin_best.pth 

# Test ViT-Base model with a custom configuration file
python individual_image_tester.py --image test_images/1000-receipt.jpg --model models/receipt_counter_vit_best.pth --config custom_config.json

# Test with different model variants (best balanced accuracy, best F1)
python individual_image_tester.py --image test_images/1000-receipt.jpg --model models/receipt_counter_vit_best_f1.pth
```

## Hardware Acceleration

This project can utilize hardware acceleration:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

The code automatically detects and uses the best available device. For Apple Silicon (M1/M2/M3) Mac users, the MPS backend is used with fallback to CPU for operations not supported by MPS.

## Model Architectures

The project compares two vision transformer architectures, with implementations available from both Hugging Face Transformers (default) and torchvision (alternative):

### Swin-Tiny Transformer
- **Architecture**: Hierarchical vision transformer using shifted windows for efficient processing
- **Backbone**: Swin-Tiny pre-trained on ImageNet
- **Parameters**: ~28 million 
- **Input Size**: 224x224 pixels
- **Output**: 6 classes (0-5 receipts)
- **Loss Function**: Cross Entropy Loss
- **Optimization**: AdamW with ReduceLROnPlateau scheduler and optional gradient clipping
- **Implementation Options**:
  - **Hugging Face (Default)**: `microsoft/swin-tiny-patch4-window7-224` with custom classifier


### ViT-Base Transformer
- **Architecture**: Standard Vision Transformer with non-overlapping patches 
- **Backbone**: ViT-Base-16 pre-trained on ImageNet
- **Parameters**: ~86 million
- **Input Size**: 224x224 pixels
- **Output**: 6 classes (0-5 receipts)
- **Loss Function**: Cross Entropy Loss
- **Optimization**: AdamW with ReduceLROnPlateau scheduler and optional gradient clipping
- **Implementation Options**:
  - **Hugging Face (Default)**: `google/vit-base-patch16-224` with custom classifier


Both models are fine-tuned using the AdamW optimizer with learning rate 5e-5, increased weight decay (0.05), and regularization techniques including BatchNorm and Dropout to prevent overfitting. The ReduceLROnPlateau scheduler with patience=2 and factor=0.5 helps stabilize training and prevent erratic behavior after convergence. Gradient clipping can be applied with the `--grad_clip` parameter when available.

## Configuration System

This project includes a centralized configuration system to make the receipt counter adaptable to changes in class distribution. This is especially important in production environments where the real-world distribution of classes may differ from the training distribution.

### Configuration File

The configuration system uses a JSON file for defining class distribution and calibration factors:

```json
{
  "class_distribution": [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
  "calibration_factors": [1.0, 1.0, 0.7, 0.6, 0.55, 0.5]
}
```

A default configuration file (`receipt_config.json`) is included, but you can create custom configurations for different deployment scenarios.

### Environment Variables

Configuration can also be provided through environment variables:

```bash
# Set class distribution
export RECEIPT_CLASS_DIST="0.4,0.2,0.2,0.1,0.1"

# Set config file path
export RECEIPT_CONFIG_PATH="/path/to/custom/config.json"
```

Environment variables take precedence over the config file. The calibration factors will be automatically derived from the class distribution.

### Using the Configuration System

All core scripts support loading configuration using the `--config` parameter:

```bash
# Training with custom configuration
python train_vit_classification.py --config custom_config.json

# Direct override of class distribution
python train_vit_classification.py --class_dist "0.25,0.25,0.2,0.1,0.1,0.1"

# Testing with custom configuration
python individual_image_tester.py --image test.jpg --model model.pth --config custom_config.json

# Creating collages with custom distribution from configuration
python create_receipt_collages.py --config custom_config.json
```

### How the Configuration System Works

The configuration system is implemented as a singleton in `config.py`:

1. When imported, it initializes with default values
2. It checks for a configuration file (`receipt_config.json` by default)
3. It checks for environment variables that might override file settings
4. Based on the class distribution, it automatically calculates:
   - Inverse weights for loss function
   - Normalized weights
   - Scaled weights for CrossEntropyLoss
   - **Bayesian calibration factors** for inference-time calibration

The simplified JSON configuration only requires specifying the class distribution:

```json
{
  "class_distribution": [0.4, 0.2, 0.2, 0.1, 0.1]
}
```

During training and inference, the system provides tensors ready for use with PyTorch:

```python
from config import get_config

# Get configuration singleton
config = get_config()

# Get class weights for loss function
weights = config.get_class_weights_tensor(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# Get automatically derived calibration factors for inference
calibration = config.get_calibration_tensor(device)
class_prior = config.get_class_prior_tensor(device)

# To understand the derivation of calibration factors
explanation = config.explain_calibration()
print(explanation)
```

The `explain_calibration()` method provides a detailed breakdown of how each calibration factor is derived from the class distribution, making the system transparent and interpretable.

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

## Transformer Module Usage

The code is designed to use Hugging Face implementations:

```python
# Transformer implementations
from transformer_swin import create_swin_transformer  # Swin-Tiny
from transformer_vit import create_vit_transformer    # ViT-Base

# Loading pre-trained models
from transformer_swin import load_swin_model
from transformer_vit import load_vit_model
```