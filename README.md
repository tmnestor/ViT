# SwinV2 Receipt Classifier

A computer vision project using the SwinV2 Transformer architecture for classifying the number of receipts in images. Supports both SwinV2-Tiny and SwinV2-Large models with a centralized configuration system.

## Purpose

This project classifies the number of receipts present in an image using the advanced SwinV2 vision transformer architecture. It's designed to:

1. Detect and count receipts in images with high accuracy
2. Provide confidence scores for each prediction
3. Handle class imbalance through robust calibration techniques
4. Achieve high performance without extensive computational requirements

The classifier is particularly useful for document processing applications, digitizing paper receipts, and automated accounting systems.

## Usage

### Setup

```bash
# Option 1: Create a conda environment
conda env create -f environment.yml
conda activate vit_env

# Option 2: Using pip with venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Generate synthetic training data
python create_receipt_collages.py --num_collages 300

# 2. Create dataset from collages
python create_collage_dataset.py --collage_dir receipt_collages --output_dir receipt_dataset

# 3. Train the model (using SwinV2-Tiny by default)
python train_swinv2_classification.py -tc receipt_dataset/train.csv -td receipt_dataset/train \
                              -vc receipt_dataset/val.csv -vd receipt_dataset/val \
                              -s 42 -d

# 4. Evaluate the model (SwinV2-Tiny)
python evaluate_swinv2_classifier.py --model models/receipt_counter_swinv2_best.pth \
                                   --test_csv receipt_dataset/test.csv \
                                   --test_dir receipt_dataset/test

# 5. Evaluate SwinV2-Large model
python evaluate_swinv2_classifier.py --model models/receipt_counter_swinv2-large_best.pth \
                                   --test_csv receipt_dataset/test.csv \
                                   --test_dir receipt_dataset/test \
                                   --model_type swinv2-large

# 6. Test a single image
python individual_image_tester.py --image receipt_collages/collage_254_3_receipts.jpg \
                                 --model models/receipt_counter_swinv2_best.pth
```

### Advanced Usage

#### Model Options

```bash
# Train with SwinV2-Tiny model (default)
python train_swinv2_classification.py --model_type swinv2

# Train with SwinV2-Large model (22K pre-trained)
python train_swinv2_classification.py --model_type swinv2-large

# Download model weights for offline use
python swinv2_model_download.py --model_type tiny
python swinv2_model_download.py --model_type large
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
# Evaluate SwinV2-Tiny model
python evaluate_swinv2_classifier.py \
  --model models/receipt_counter_swinv2_best.pth \
  --test_csv receipt_dataset/test.csv \
  --test_dir receipt_dataset/test \
  [--no-calibration] # without calibration

# Evaluate SwinV2-Large model
python evaluate_swinv2_classifier.py \
  --model models/receipt_counter_swinv2-large_best.pth \
  --test_csv receipt_dataset/test.csv \
  --test_dir receipt_dataset/test

# Evaluate in binary mode (0 vs 1+ receipts)
python evaluate_swinv2_classifier.py --model models/receipt_counter_swinv2_best.pth --binary
```

#### Testing Individual Images

```bash
# Test with SwinV2-Tiny model
python individual_image_tester.py --image receipt_collages/collage_254_3_receipts.jpg \
                               --model models/receipt_counter_swinv2_best.pth

# Test with SwinV2-Large model
python individual_image_tester.py --image receipt_collages/collage_254_3_receipts.jpg \
                               --model models/receipt_counter_swinv2-large_best.pth \
                               --model-type swinv2-large
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
    "learning_rate": 5e-5,
    "backbone_lr_multiplier": 0.1,
    "epochs": 30,
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
  - `swinv2_model_download.py` - Download pre-trained models

- **Dataset & Training**
  - `datasets.py` - Dataset classes
  - `training_utils.py` - Training and evaluation utilities
  - `train_swinv2_classification.py` - Training script
  - `evaluate_swinv2_classifier.py` - Evaluation script

- **Data Generation**
  - `create_receipt_collages.py` - Generate synthetic receipt collages
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


poch 3/20, Train Loss: 0.5786, Val Loss: 0.4934, Accuracy: 100.00%, Balanced Accuracy: 100.00%, F1 Macro: 100.00%
Saved new best model with F1 Macro: 1.0000
Epoch 4/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:31<00:00,  4.53s/it, loss=0.48]
Epoch 4/20, Train Loss: 0.5097, Val Loss: 0.5698, Accuracy: 80.00%, Balanced Accuracy: 77.94%, F1 Macro: 75.49%
Saved new best model with F1 Macro: 0.7549
EarlyStopping: No improvement for 1 epochs. Best value: 1.0000, Current value: 0.7549
Epoch 5/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:30<00:00,  4.41s/it, loss=0.467]
Epoch 5/20, Train Loss: 0.4529, Val Loss: 0.3948, Accuracy: 95.56%, Balanced Accuracy: 93.87%, F1 Macro: 94.01%
Saved new best model with F1 Macro: 0.9401
EarlyStopping: No improvement for 2 epochs. Best value: 1.0000, Current value: 0.9401
Epoch 6/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:30<00:00,  4.37s/it, loss=0.52]
Epoch 6/20, Train Loss: 0.4556, Val Loss: 0.3493, Accuracy: 97.78%, Balanced Accuracy: 98.04%, F1 Macro: 97.03%
Saved new best model with F1 Macro: 0.9703
EarlyStopping: No improvement for 3 epochs. Best value: 1.0000, Current value: 0.9703
EarlyStopping: Stopping training after 3 epochs with no improvement
Early stopping triggered after 6 epochs

Loading best model for final evaluation...
Loading model from models/receipt_counter_swinv2_best.pth as a SwinV2 model
Successfully loaded model with strict=True

Best Model Results:
Accuracy: 97.78%
F1 Macro: 97.03%