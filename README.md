# Vision Transformer (ViT) Receipt Counter

A computer vision project comparing Vision Transformer architectures (Swin-Tiny and ViT-Base) for counting receipts in images.

## Project Overview

This project compares the Swin-Tiny and ViT-Base vision transformer architectures to determine which can more accurately count the number of receipts in an image. Both models use a regression approach to predict a continuous count value, which can be rounded to get the final receipt count.

## Setup

### Option 1: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
  - `receipt_counter.py` - Swin-Tiny model implementation for receipt counting
  - `vit_counter.py` - ViT-Base model implementation for receipt counting
  - `receipt_processor.py` - Image preprocessing utilities

- **Data Generation & Preparation:**
  - `create_realistic_collages.py` - Generate synthetic receipt collages on table surfaces
  - `prepare_collage_dataset.py` - Prepare datasets from collage images
  - `download_test_images.py` - Utility to download/process SRD receipt dataset

- **Training & Evaluation:**
  - `train_swin_counter.py` - Train the Swin-Tiny receipt counter model
  - `train_vit_counter.py` - Train the ViT-Base receipt counter model
  - `evaluate_swin_counter.py` - Evaluate Swin-Tiny model performance with metrics
  - `evaluate_vit_counter.py` - Evaluate ViT-Base model performance with metrics
  - `demo.py` - Process individual images through the trained model

- **Additional Utilities:**
  - `torchvision_demo.py` - Demo of pre-trained ViT from torchvision
  - `api.py` - Flask API for receipt counting as a service
  - `batch_processor.py` - Batch processing utilities
  - `test_images_demo.py` - Testing with multiple sample images

## Workflow

### 1. Generate Synthetic Training Data

You can generate realistic collages of receipts on various table surfaces using the collage generator:

```bash
# Generate 300 collages with 0-5 receipts based on default probability distribution
python create_receipt_collages.py --num_collages 300

# Generate with specific probability distribution for receipt counts
# Format: p0,p1,p2,p3,p4,p5 where pN is probability of having N receipts
python create_receipt_collages.py --num_collages 300 --count_probs 0.3,0.2,0.2,0.1,0.1,0.1

# Full control over output
python create_receipt_collages.py --num_collages 200 \
  --canvas_width 1600 --canvas_height 1200 \
  --count_probs 0.3,0.2,0.2,0.1,0.1,0.1
```

The collages will be saved in a `receipt_collages` directory with filenames indicating how many receipts are in each image. The current version places receipts in the center of the collage without overlapping and limits rotation to ±10 degrees for better readability.

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
# Train Swin-Tiny model with prepared dataset
python train_swin_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                           --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                           --epochs 15 --batch_size 32 --output_dir models

# Train ViT-Base model with prepared dataset
python train_vit_classification.py --train_csv receipt_dataset/train.csv --train_dir receipt_dataset/train \
                          --val_csv receipt_dataset/val.csv --val_dir receipt_dataset/val \
                          --epochs 15 --batch_size 32 --output_dir models
```

Training generates evaluation metrics, plots, and saves models to the `models` directory for both architectures.

### 5. Evaluate the Models

```bash
# Evaluate Swin-Tiny model on validation set
python evaluate_swin_counter.py --model models/receipt_counter_best.pth \
                                 --test_csv receipt_dataset/val.csv \
                                 --test_dir receipt_dataset/val \
                                 --output_dir evaluation/swin_tiny

# Evaluate ViT-Base model on validation set
python evaluate_vit_counter.py --model models/receipt_counter_vit_best.pth \
                             --test_csv receipt_dataset/val.csv \
                             --test_dir receipt_dataset/val \
                             --output_dir evaluation/vit_base
```

The evaluation scripts generate detailed metrics and visualizations in their respective output directories, allowing for direct comparison between the two models.

### 6. Test on Individual Images

```bash
# Test Swin-Tiny model on a single image
python demo.py --image test_images/sample_receipt_1000.jpg --mode local --model models/receipt_counter_best.pth

# Test ViT-Base model on a single image (requires modifying demo.py to use ViTReceiptCounter)
python demo.py --image test_images/sample_receipt_1000.jpg --mode local --model models/receipt_counter_vit_best.pth
```

Note: To use the ViT model with the demo script, you'll need to modify `demo.py` to import and use `ViTReceiptCounter` from `vit_counter.py`.

### 7. Run the API (Optional)

```bash
# Start the API service
python api.py
```

By default, the API runs on http://localhost:5000 with the following endpoint:
- `POST /count_receipts` - Upload an image for receipt counting

Example API request:
```bash
curl -X POST -F "file=@test_images/sample_receipt_1000.jpg" http://localhost:5000/count_receipts
```

## Hardware Acceleration

This project can utilize hardware acceleration:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

The code automatically detects and uses the best available device. For Apple Silicon (M1/M2/M3) Mac users, the MPS backend is used with fallback to CPU for operations not supported by MPS.

## Model Architectures

The project compares two vision transformer architectures:

### Swin-Tiny Transformer
- **Architecture**: Hierarchical vision transformer using shifted windows for efficient processing
- **Backbone**: Swin-Tiny pre-trained on ImageNet
- **Parameters**: ~28 million 
- **Input Size**: 224x224 pixels
- **Output**: Single regression value (receipt count)
- **Loss Function**: Mean Squared Error (MSE)

### ViT-Base Transformer
- **Architecture**: Standard Vision Transformer with non-overlapping patches 
- **Backbone**: ViT-Base-16 pre-trained on ImageNet
- **Parameters**: ~86 million
- **Input Size**: 224x224 pixels
- **Output**: Single regression value (receipt count)
- **Loss Function**: Mean Squared Error (MSE)

Both models are fine-tuned using the AdamW optimizer with a cosine annealing learning rate schedule, increased weight decay (0.05), and regularization techniques including BatchNorm and Dropout to prevent overfitting.

## Credits

This project uses the following vision transformer architectures:

- **Swin Transformer**: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

- **Vision Transformer (ViT)**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR).