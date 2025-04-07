# SwinV2 Receipt Classifier

A computer vision project using the SwinV2 Transformer architecture for classifying the number of receipts in images. Supports both SwinV2-Tiny and SwinV2-Large models with a centralized configuration system.

## Purpose

This project classifies the number of receipts present in an image using the advanced SwinV2 vision transformer architecture. It's designed to:

1. Detect and count receipts in images with high accuracy
2. Provide confidence scores for each prediction
3. Handle class imbalance through robust calibration techniques
4. Distinguish between receipts and non-receipt documents (Australian tax documents)
5. Process images in both portrait and landscape orientations
6. Achieve high performance without extensive computational requirements

The classifier is particularly useful for document processing applications, digitizing paper receipts, and automated accounting systems. The project includes tools for generating synthetic receipts and anonymized Australian tax documents for training.

## Usage

### Setup

```bash
# Option 1: Create a conda environment
conda env create -f environment.yml
conda activate vit_env

# Option 2: Using pip with venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Note: requirements.txt includes all the necessary packages 
# with specific versions for reproducible environment setup
```

### Quick Start

```bash
# 1. Download model weights for offline use
python huggingface_model_download.py --model_name "microsoft/swinv2-large-patch4-window12-192-22k" --output_dir /path/to/models/swinv2-large

# 2. Generate synthetic receipts with stapled payment receipts
python create_synthetic_receipts.py --num_collages 1000 --count_probs "0.3,0.3,0.2,0.1,0.1" --output_dir synthetic_receipts --stapled_ratio 0.3

# 3. Create dataset from synthetic receipts
python create_collage_dataset.py --collage_dir synthetic_receipts --output_dir receipt_dataset

# 4. Train the SwinV2-Large model with offline weights
python train_swinv2_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                              -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                              -e 20 -b 8 -o /path/to/output/models -s 42 -d \
                              --model_type swinv2-large --offline \
                              --pretrained_model_dir /path/to/models/swinv2-large

# 5. Evaluate the trained model
python evaluate_swinv2_classifier.py --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
                                   --test_csv receipt_dataset/test.csv \
                                   --test_dir receipt_dataset/test \
                                   --model_type swinv2-large

# 6. Test a single image with the trained model
python individual_image_tester.py --image synthetic_receipts/synthetic_001_2_receipts.jpg \
                                 --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
                                 --model_type swinv2-large
```



#### Model Options

```bash
# Train with SwinV2-Tiny model (default)
python train_swinv2_classification.py --model_type swinv2

# Train with SwinV2-Large model (22K pre-trained)
python train_swinv2_classification.py --model_type swinv2-large

# Production Model Management
# Download models for offline use (never store in source code root)
python huggingface_model_download.py --model_name "google/vit-base-patch16-224" --output_dir /path/to/models/vit
python huggingface_model_download.py --model_name "facebook/deit-base-patch16-224" --output_dir /path/to/models/deit
python huggingface_model_download.py --model_name "microsoft/swinv2-tiny-patch4-window8-256" --output_dir /path/to/models/swinv2-tiny
python huggingface_model_download.py --model_name "microsoft/swinv2-large-patch4-window12-192-22k" --output_dir /path/to/models/swinv2-large

# Train using pre-downloaded models
# ViT model example
python train_vit_classification.py --model_name "google/vit-base-patch16-224" --offline \
                                  --pretrained_model_dir /path/to/models/vit

# SwinV2 model examples
python train_swinv2_classification.py --model_type swinv2 --offline \
                                    --pretrained_model_dir /path/to/models/swinv2-tiny

python train_swinv2_classification.py --model_type swinv2-large --offline \
                                    --pretrained_model_dir /path/to/models/swinv2-large
```

#### Training Options

```bash
# Train with custom class distribution (adjust for different dataset imbalance)
python train_swinv2_classification.py --class_dist "0.4,0.3,0.3"

# Train with custom gradient clipping and differential learning rates
python train_swinv2_classification.py --lr 5e-5 --backbone_lr_multiplier 0.1 --grad_clip 2.0

# Resume training from a checkpoint
python train_swinv2_classification.py -r models/receipt_counter_swinv2_best.pth

# Dry run to validate configuration
python train_swinv2_classification.py --dry-run -s 42
```

#### Evaluation Options

```bash
# Standard evaluation of a trained model
python evaluate_swinv2_classifier.py \
  --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
  --test_csv receipt_dataset/test.csv \
  --test_dir receipt_dataset/test \
  --model_type swinv2-large

# Evaluate with optional parameters
python evaluate_swinv2_classifier.py \
  --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
  --test_csv receipt_dataset/test.csv \
  --test_dir receipt_dataset/test \
  --output_dir evaluation/results \
  --model_type swinv2-large \
  [--no-calibration] # without calibration

# Evaluate in binary mode (0 vs 1+ receipts)
python evaluate_swinv2_classifier.py \
  --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
  --test_csv receipt_dataset/test.csv \
  --test_dir receipt_dataset/test \
  --model_type swinv2-large \
  --binary
```

#### Testing Individual Images

```bash
# Test a synthetic receipt with 2 receipts
python individual_image_tester.py --image synthetic_receipts/synthetic_001_2_receipts.jpg \
                               --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
                               --model_type swinv2-large

# Test a synthetic receipt with 0 receipts (tax document)
python individual_image_tester.py --image synthetic_receipts/synthetic_042_0_receipts.jpg \
                               --model /path/to/output/models/receipt_counter_swinv2-large_best.pth \
                               --model_type swinv2-large
```

## Theory

### SwinV2 Architecture

SwinV2 is an advanced vision transformer developed by Microsoft Research that addresses key limitations in scaling vision models:

1. **Residual Post-Norm with Cosine Attention**: Improves training stability for large-scale models
2. **Log-Spaced Continuous Position Bias**: Enables better transfer between different image resolutions
3. **Self-Supervised Pre-training (SimMIM)**: Reduces the need for labeled data

The project supports two SwinV2 variants:

**SwinV2-Tiny**:
- Patch size: 4 pixels
- Window size: 8
- Input resolution: 256x256
- Embedding dimension: 96
- Layer depths: [2, 2, 6, 2]
- Attention heads: [3, 6, 12, 24]

**SwinV2-Large**:
- Patch size: 4 pixels
- Window size: 12
- Input resolution: 192x192
- Embedding dimension: 192
- Layer depths: [2, 2, 18, 2]
- Attention heads: [6, 12, 24, 48]
- Pre-trained on ImageNet-22K

### Classification Metrics

Our evaluation uses multiple metrics to assess model performance:

- **Accuracy**: Simple percentage of correct predictions
- **Balanced Accuracy**: Average recall across all classes, robust to class imbalance
- **F1-Macro Score**: Harmonic mean of precision and recall, averaged equally across classes

Balanced accuracy and F1-macro are particularly important in our imbalanced dataset where some receipt counts are less frequent.

### Class Weighting and Calibration

This project implements a principled approach to handle class imbalance:

1. **Training-time Weighting**: Applies inverse frequency weighting during training:

   $$\text{Normalized Weight}_i = \frac{\frac{1}{p_i}}{\sum_{j=1}^{n} \frac{1}{p_j}} \times n$$

2. **Bayesian Calibration**: Applies principled correction during inference:

   $$\text{Calibration Factor}_i = p_i \times \sqrt{\frac{p_{\text{ref}}}{p_i}}$$

   $$\text{Calibrated Probability}_i = \frac{\text{Raw Probability}_i \times \text{Calibration Factor}_i \times p_i}{\sum_{j=1}^{n} \text{Raw Probability}_j \times \text{Calibration Factor}_j \times p_j}$$

This two-stage approach ensures the model learns effectively from minority classes while producing properly calibrated predictions that match the true class distribution.

## Configuration System

The project uses a centralized configuration system for all parameters:

```python
# Default values in config.py
DEFAULT_MODEL_PARAMS = {
    # Image parameters
    "image_size": 256,  # SwinV2-Tiny uses 256x256 images by default
    
    # Model selection
    "model_type": "swinv2",  # "swinv2" or "swinv2-large"
    
    # Training parameters
    "batch_size": 8,
    "learning_rate": 5e-4,
    "backbone_lr_multiplier": 0.02,
    "epochs": 20,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    
    # Reproducibility parameters
    "random_seed": 42,
    "deterministic_mode": True,
}
```

### Configuration Priority

Parameters are determined in the following order:

1. Command-line arguments (highest priority)
2. Environment variables (e.g., `RECEIPT_BATCH_SIZE`, `RECEIPT_LEARNING_RATE`)  
3. Config JSON file (if specified with `--config`)
4. Default values in `config.py` (lowest priority)

## Project Structure

- **Core Model Files**
  - `model_factory.py` - Factory pattern for creating and loading models
  - `config.py` - Centralized configuration system
  - `huggingface_model_download.py` - Download any pre-trained HuggingFace model for offline use

- **Dataset & Training**
  - `datasets.py` - Dataset classes
  - `training_utils.py` - Training and evaluation utilities
  - `train_swinv2_classification.py` - Training script
  - `evaluate_swinv2_classifier.py` - Evaluation script

- **Data Generation**
  - `create_receipt_collages.py` - Generate synthetic receipt collages with portrait/landscape orientations
  - `create_synthetic_receipts.py` - Generate fully synthetic receipts (100 individual samples) with stapled payment receipts
    - New `--stapled_ratio` parameter controls proportion of receipts with payment receipts stapled on front
    - Leverages existing `generate_payment_receipt()` function to create smaller payment slips
    - Stapled payment receipts create more realistic training data matching real-world tax documents
  - `create_tax_documents.py` - Generate anonymized Australian tax documents for 0-receipt examples
  - `create_collage_dataset.py` - Create datasets from collages
  - `create_rectangle_dataset.py` - Generate simplified rectangle-based dataset

- **Utilities**
  - `device_utils.py` - Hardware acceleration utilities
  - `reproducibility.py` - Seed control for consistent results
  - `individual_image_tester.py` - Process individual images
  - `test_images_demo.py` - Test multiple images

## References

1. **SwinV2 Transformer**: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

2. **Calibration Techniques**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1321-1330).

3. **Bayesian Calibration**: Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., & Flach, P. (2019). Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. In Advances in Neural Information Processing Systems (pp. 12316-12326).

4. **Transfer Learning with Differential Learning Rates**: Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (pp. 328-339).

5. **Hugging Face Transformers**: Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. (2020). Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45).

6. **Weight Normalization**: Salimans, T., & Kingma, D. P. (2016). Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems (pp. 901-909).

7. **Jeffreys Prior for Calibration**: Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 186(1007), 453-461.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Planned Modifications

## Stapled Receipt Modification Plan

The `create_synthetic_receipts.py` script will be modified to incorporate payment receipts stapled on top of regular receipts:

1. Update the `create_receipt_collage()` function to accept a new `stapled_ratio` parameter
   - This parameter controls the proportion of receipts that will have payment receipts stapled on top

2. Modify the receipt generation process to:
   - For each receipt in a collage, check against the `stapled_ratio` probability
   - If selected for stapling, create a payment receipt using the existing `generate_payment_receipt()` function
   - Position the payment receipt partially overlapping the main receipt
   - Apply a slight rotation to the payment receipt for realism
   - Add staple marks at the overlap points

3. Add a new CLI parameter to `main()`:
   ```python
   parser.add_argument("--stapled_ratio", type=float, default=0.0,
                     help="Proportion of receipts with payment receipts stapled on top (0.0-1.0)")
   ```

4. Pass the `stapled_ratio` parameter from command line to the `create_receipt_collage()` function

5. Update documentation with usage examples for the new parameter

The implementation will maintain the correct receipt count for model training purposes, while creating more realistic training data that mimics real-world tax documents where payment receipts are often stapled to the main receipt.


<!-- python huggingface_model_download.py --model_name "microsoft/swinv2-large-patch4-window12-192-22k" --output_dir /Users/tod/PretrainedLLM/swin_large

python train_swinv2_classification.py --model_type swinv2 --offline \
                                    --pretrained_model_dir /Users/tod/PretrainedLLM/swin_large



                                    python train_swinv2_classification.py --model_type swinv2 --offline \
                                    --pretrained_model_dir ../swin_large -->