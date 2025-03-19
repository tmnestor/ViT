# Comparative Analysis: ViT-Base vs Swin-Tiny for Receipt Counting in Scanned Images

## Executive Summary
This report compares two vision transformer architectures—ViT-Base (86M parameters) and Swin-Tiny (28M parameters)—for the specific application of receipt counting in scanned documents. Both models were trained and evaluated on the same dataset of scanned receipt images. Test results show that **Swin-Tiny outperforms ViT-Base** across multiple metrics, with better accuracy, lower error rates, faster inference times, and smaller memory footprint. The hierarchical design of Swin-Tiny appears particularly well-suited for document analysis tasks compared to the standard Vision Transformer architecture.

## Model Architectures

### ViT-Base (86M parameters)
- **Architecture**: Standard Vision Transformer with 12 layers, 768 hidden dimension
- **Patch Size**: 16×16 pixels
- **Attention Mechanism**: Global self-attention across all patches
- **Positional Encoding**: 1D positional embeddings
- **PyTorch Implementation**: `torchvision.models.vit_b_16` (pretrained on ImageNet-1K)

### Swin-Tiny (28M parameters)
- **Architecture**: Hierarchical Transformer with 4 stages, shifted windows
- **Patch Size**: 4×4 pixels initially, with hierarchical merging
- **Attention Mechanism**: Local self-attention within windows, with cross-window connections
- **Positional Encoding**: Relative position bias
- **PyTorch Implementation**: `torchvision.models.swin_t` (pretrained on ImageNet-1K)

## Experimental Setup

### Dataset
- **Source**: Synthetic collages created from SRD (Scanned Receipt Dataset)
- **Size**: 2000 collage images (1600 training, 400 validation)
- **Distribution**: Balanced across 0-4 receipts per image
- **Images**: 224×224 pixels, RGB format
- **Preprocessing**: Resizing, normalization with ImageNet statistics

### Training Configuration
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with weight decay (0.05)
- **Learning Rate**: 3e-5 with cosine annealing
- **Batch Size**: 16
- **Epochs**: 30 (with early stopping)
- **Regularization**: BatchNorm, Dropout (0.3)
- **Hardware**: MPS acceleration (Apple Silicon)

## Performance Evaluation Results

### Key Metrics Comparison

| Metric | Swin-Tiny | ViT-Base | Difference |
|--------|-----------|----------|------------|
| MAE (Mean Absolute Error) | 0.434 | 0.512 | **17.9% better** |
| RMSE (Root Mean Square Error) | 0.569 | 0.643 | **13.0% better** |
| Exact Match Rate | 63.8% | 56.5% | **+7.3% points** |
| Average Inference Time | ~12ms | ~25ms | **2.1x faster** |
| Model Size | 110MB | 340MB | **3.1x smaller** |

### Error Analysis by Receipt Count

| Receipt Count | Swin-Tiny MAE | ViT-Base MAE | Swin Advantage |
|---------------|---------------|--------------|----------------|
| 0 receipts | 0.208 | 0.359 | **42.1% better** |
| 1 receipt | 0.375 | 0.402 | **6.7% better** |
| 2 receipts | 0.537 | 0.507 | 5.9% worse |
| 3 receipts | 0.488 | 0.561 | **13.0% better** |
| 4 receipts | 0.887 | 1.069 | **17.0% better** |

### Prediction Bias Analysis

| Receipt Count | Swin-Tiny Bias | ViT-Base Bias | Observation |
|---------------|----------------|---------------|-------------|
| 0 receipts | -0.208 | -0.359 | Both overpredict, ViT worse |
| 1 receipt | +0.061 | -0.006 | Swin slightly underpredicts |
| 2 receipts | -0.105 | -0.242 | Both overpredict, ViT worse |
| 3 receipts | -0.299 | -0.489 | Both overpredict, ViT worse |
| 4 receipts | -0.883 | -1.069 | Both underpredict, ViT worse |

### Visual Performance Comparison

Qualitative analysis of the visualization plots reveals:

1. **Prediction Distribution**:
   - Swin-Tiny shows tighter clustering around the perfect prediction line
   - ViT-Base shows more scattered predictions, especially at higher receipt counts

2. **Error Distribution**:
   - Swin-Tiny has a narrower error distribution centered closer to zero
   - ViT-Base shows a wider error distribution with a stronger negative skew

3. **Confusion Matrix**:
   - Swin-Tiny has higher concentration on the diagonal (correct predictions)
   - ViT-Base shows more off-diagonal spread, especially underpredicting high counts

## Analysis of Results

### Strengths of Swin-Tiny

1. **Superior Accuracy**: Consistently outperforms ViT-Base in overall metrics and for most count categories.

2. **Better Performance at Extremes**: Particularly effective at correctly counting zero receipts (42.1% better) and handling higher receipt counts (17% better at 4 receipts).

3. **Local Feature Capture**: The windowed attention mechanism likely handles local spatial features better, which is crucial for identifying individual receipts.

4. **Computational Efficiency**: 2.1x faster inference with 3.1x smaller model size makes it much more practical for deployment.

5. **Less Systematic Bias**: While both models show predictive biases, Swin-Tiny's bias is consistently smaller across almost all count categories.

### Weaknesses of ViT-Base

1. **Global Attention Limitations**: The global attention mechanism may dilute focus on local features needed to distinguish individual receipts.

2. **Higher Computational Overhead**: Requires more memory and computation for similar or worse results.

3. **Overprediction at Zero**: Shows a strong tendency to predict receipts when none are present (bias of -0.359).

4. **Underprediction at High Counts**: Severely underpredicts when many receipts are present (bias of -1.069 at 4 receipts).

5. **Parameter Inefficiency**: Despite having 3x more parameters, delivers worse performance, suggesting architecture mismatch with the task.

### Exception Cases

Interestingly, ViT-Base slightly outperforms Swin-Tiny for images with exactly 2 receipts (MAE of 0.507 vs 0.537). This could indicate:

1. A potential sweet spot for ViT's global attention when dealing with a moderate number of objects
2. Possible data distribution characteristics specific to the 2-receipt images
3. Random variation that might normalize with a larger test set

## Implementation Recommendations

### For Receipt Counting Applications

1. **Model Selection**:
   - **Primary Recommendation**: Swin-Tiny for its superior accuracy and efficiency
   - **Alternative**: ViT-Base only if computational resources are not a concern and extra parameters might help with transfer to other document tasks

2. **Training Strategy**:
   - **Swin-Tiny**: Learning rate of 3e-5, 15-20 epochs typically sufficient
   - **ViT-Base**: Lower learning rate (1e-5), longer training (25-30 epochs) if used
   - Both benefit from weight decay of 0.05 and dropout of 0.3

3. **Data Augmentation Emphasis**:
   - Focus on generating examples with higher receipt counts (3-4+) to address underprediction bias
   - Include more empty (0 receipt) examples for ViT training to reduce overprediction

4. **Post-processing Adjustments**:
   - Apply count-specific bias correction based on the error analysis table
   - For Swin-Tiny: Add ~0.88 to predictions of 4 receipts
   - For ViT-Base: Add ~1.07 to predictions of 4 receipts

## Conclusion and Recommendation

For the specific application of receipt counting in scanned images, **Swin-Tiny** is clearly recommended over ViT-Base based on empirical evaluation. The advantages of Swin-Tiny include:

1. **Better accuracy** across most receipt count categories (17.9% lower MAE overall)
2. **Faster inference time** (2.1x speedup) enabling real-time applications
3. **Lower memory requirements** (3.1x smaller) allowing deployment in more constrained environments
4. **Better handling of spatial hierarchies** likely due to its window-based attention mechanism
5. **More stable predictions** with less systematic bias

While ViT-Base offers marginally better performance for images with exactly 2 receipts, this isolated advantage doesn't outweigh Swin-Tiny's comprehensive benefits. For real-world applications where efficiency and accuracy are both important, Swin-Tiny provides the optimal balance.

## Software Stack

### Common Software Requirements
- **Framework**: PyTorch 2.0+
- **Libraries**:
  - torchvision 0.15+
  - timm 0.9+ (optional, for advanced augmentations)
  - albumentations 1.3+ (for document-specific preprocessing)
  - opencv-python 4.7+ (for image preprocessing)
- **Hardware**: CUDA-compatible GPU or Apple Silicon with MPS

### Model-Specific Requirements
- **ViT-Base**: Higher memory footprint (~340MB model size)
- **Swin-Tiny**: Lower memory footprint (~110MB model size)

## References

### Model Papers
1. **Vision Transformer (ViT)**:
   - Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations (ICLR)*.
   - [Paper Link](https://arxiv.org/abs/2010.11929)

2. **Swin Transformer**:
   - Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *IEEE/CVF International Conference on Computer Vision (ICCV)*.
   - [Paper Link](https://arxiv.org/abs/2103.14030)

### Implementation Resources
1. **PyTorch Hub**:
   - ViT-Base-16 model: https://download.pytorch.org/models/vit_b_16-c867db91.pth
   - Swin-Tiny model: https://download.pytorch.org/models/swin_t-704ceda3.pth
   - Documentation: https://pytorch.org/vision/stable/models.html

2. **Torchvision Documentation**:
   - ViT: https://pytorch.org/vision/stable/models/vision_transformer.html
   - Swin: https://pytorch.org/vision/stable/models/swin_transformer.html