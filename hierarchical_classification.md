# Hierarchical Classification for Imbalanced Receipt Counting

This document outlines the pseudocode for implementing a hierarchical classification approach to handle the imbalanced receipt counting task (0-5 receipts).

## 1. Hierarchical Structure

For the receipt counting problem, we'll use a 2-level hierarchy:

```
Level 1: Binary classifier (0 vs 1+ receipts)
├── If 0: FINAL PREDICTION = 0
└── If 1+: Proceed to Level 2
    
Level 2: Binary classifier (1 vs 2+ receipts)
├── If 1: FINAL PREDICTION = 1
└── If 2+: Run standard multiclass classifier for {2, 3, 4, 5}
    └── FINAL PREDICTION = {2, 3, 4, or 5}
```

## 2. Data Preparation

```python
def prepare_hierarchical_datasets(dataset_path, csv_file):
    """
    Prepare datasets for each level of the hierarchy.
    """
    # Load original dataset
    df = pd.read_csv(os.path.join(dataset_path, csv_file))
    
    # Level 1: 0 vs 1+ receipts
    df_level1 = df.copy()
    df_level1['label'] = df_level1['receipt_count'].apply(lambda x: 0 if x == 0 else 1)
    
    # Level 2: 1 vs 2+ receipts (filter out 0 receipts)
    df_level2 = df[df['receipt_count'] > 0].copy()
    df_level2['label'] = df_level2['receipt_count'].apply(lambda x: 0 if x == 1 else 1)
    
    # Multiclass data for 2+ receipts (for final classification)
    df_multiclass = df[df['receipt_count'] > 1].copy()
    # Keep original label (2, 3, 4, 5)
    
    # Create CSV files for each level
    df_level1.to_csv(os.path.join(dataset_path, 'level1_train.csv'), index=False)
    df_level2.to_csv(os.path.join(dataset_path, 'level2_train.csv'), index=False)
    df_multiclass.to_csv(os.path.join(dataset_path, 'multiclass_train.csv'), index=False)
    
    return df_level1, df_level2, df_multiclass
```

## 3. Training Pipeline

```python
def train_hierarchical_model(base_path, train_path, val_path):
    """
    Train each level of the hierarchical model.
    """
    # Create output directory structure
    os.makedirs(os.path.join(base_path, 'level1'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'level2'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'multiclass'), exist_ok=True)
    
    # Prepare datasets for each level
    prepare_hierarchical_datasets(train_path, 'train.csv')
    prepare_hierarchical_datasets(val_path, 'val.csv')
    
    # Level 1: 0 vs 1+ receipts (binary classification)
    train_level1_model(
        model_type="vit",  # or "swin"
        train_csv=os.path.join(train_path, 'level1_train.csv'),
        val_csv=os.path.join(val_path, 'level1_val.csv'),
        train_dir=os.path.join(train_path, 'train'),
        val_dir=os.path.join(val_path, 'val'),
        output_dir=os.path.join(base_path, 'level1'),
        batch_size=16,
        epochs=30,
        learning_rate=1e-4,
        backbone_lr_multiplier=0.1,
        num_classes=2,  # Binary classification
        augment_train=True
    )
    
    # Level 2: 1 vs 2+ receipts (binary classification)
    train_level2_model(
        model_type="vit",  # or "swin"
        train_csv=os.path.join(train_path, 'level2_train.csv'),
        val_csv=os.path.join(val_path, 'level2_val.csv'),
        train_dir=os.path.join(train_path, 'train'),
        val_dir=os.path.join(val_path, 'val'),
        output_dir=os.path.join(base_path, 'level2'),
        batch_size=16,
        epochs=30,
        learning_rate=1e-4,
        backbone_lr_multiplier=0.1,
        num_classes=2,  # Binary classification
        augment_train=True
    )
    
    # Multiclass model for 2-5 receipts
    train_multiclass_model(
        model_type="vit",  # or "swin"
        train_csv=os.path.join(train_path, 'multiclass_train.csv'),
        val_csv=os.path.join(val_path, 'multiclass_val.csv'),
        train_dir=os.path.join(train_path, 'train'),
        val_dir=os.path.join(val_path, 'val'),
        output_dir=os.path.join(base_path, 'multiclass'),
        batch_size=16,
        epochs=30,
        learning_rate=1e-4,
        backbone_lr_multiplier=0.1,
        num_classes=4,  # 2, 3, 4, 5 receipts
        class_weights=[1.0, 1.0, 1.0, 1.0],  # Can be adjusted based on class distribution
        augment_train=True
    )
```

The functions `train_level1_model`, `train_level2_model`, and `train_multiclass_model` would be similar to your existing training functions but adapted to each specific binary/multiclass task.

## 4. Inference Pipeline

```python
def hierarchical_predict(image_path, model_base_path, model_type="vit", enhance=True):
    """
    Run hierarchical inference on an image.
    
    Args:
        image_path: Path to the image
        model_base_path: Base path containing level1, level2, multiclass model folders
        model_type: "vit" or "swin"
        enhance: Whether to enhance the image before processing
        
    Returns:
        predicted_count: Final receipt count prediction (0-5)
        confidence: Overall confidence score
        confidences: Dict of confidences at each level
    """
    # Get device
    device = get_device()
    
    # Enhance image if requested
    processor = ReceiptProcessor()
    if enhance:
        try:
            processor.enhance_scan_quality(image_path, "enhanced_scan.jpg")
            processed_image_path = "enhanced_scan.jpg"
        except:
            processed_image_path = image_path
    else:
        processed_image_path = image_path
    
    # Preprocess image
    img_tensor = processor.preprocess_image(processed_image_path).to(device)
    
    # Level 1: 0 vs 1+ receipts
    level1_model_path = os.path.join(model_base_path, 'level1', f'receipt_counter_{model_type}_best.pth')
    level1_model = ModelFactory.load_model(level1_model_path, model_type=model_type)
    level1_model = level1_model.to(device)
    level1_model.eval()
    
    with torch.no_grad():
        level1_outputs = level1_model(img_tensor)
        if hasattr(level1_outputs, 'logits'):
            level1_logits = level1_outputs.logits
        else:
            level1_logits = level1_outputs
        level1_probs = torch.nn.functional.softmax(level1_logits, dim=1)
        level1_prediction = torch.argmax(level1_probs, dim=1).item()
        level1_confidence = level1_probs[0, level1_prediction].item()
    
    # Store confidences at each level
    confidences = {
        'level1': {
            'has_receipts': level1_probs[0, 1].item(),
            'no_receipts': level1_probs[0, 0].item()
        }
    }
    
    # If prediction is 0 receipts, we're done
    if level1_prediction == 0:
        return 0, level1_confidence, confidences
    
    # Level 2: 1 vs 2+ receipts
    level2_model_path = os.path.join(model_base_path, 'level2', f'receipt_counter_{model_type}_best.pth')
    level2_model = ModelFactory.load_model(level2_model_path, model_type=model_type)
    level2_model = level2_model.to(device)
    level2_model.eval()
    
    with torch.no_grad():
        level2_outputs = level2_model(img_tensor)
        if hasattr(level2_outputs, 'logits'):
            level2_logits = level2_outputs.logits
        else:
            level2_logits = level2_outputs
        level2_probs = torch.nn.functional.softmax(level2_logits, dim=1)
        level2_prediction = torch.argmax(level2_probs, dim=1).item()
        level2_confidence = level2_probs[0, level2_prediction].item()
    
    # Store level 2 confidences
    confidences['level2'] = {
        'one_receipt': level2_probs[0, 0].item(),
        'multiple_receipts': level2_probs[0, 1].item()
    }
    
    # If prediction is 1 receipt, we're done
    if level2_prediction == 0:  # 0 means class "1 receipt" in level 2
        return 1, level2_confidence, confidences
    
    # If 2+ receipts, use multiclass model for final prediction
    multiclass_model_path = os.path.join(model_base_path, 'multiclass', f'receipt_counter_{model_type}_best.pth')
    multiclass_model = ModelFactory.load_model(multiclass_model_path, model_type=model_type)
    multiclass_model = multiclass_model.to(device)
    multiclass_model.eval()
    
    with torch.no_grad():
        multiclass_outputs = multiclass_model(img_tensor)
        if hasattr(multiclass_outputs, 'logits'):
            multiclass_logits = multiclass_outputs.logits
        else:
            multiclass_logits = multiclass_outputs
        multiclass_probs = torch.nn.functional.softmax(multiclass_logits, dim=1)
        multiclass_prediction = torch.argmax(multiclass_probs, dim=1).item()
        multiclass_confidence = multiclass_probs[0, multiclass_prediction].item()
    
    # Store multiclass confidences
    confidences['multiclass'] = {
        '2_receipts': multiclass_probs[0, 0].item(), 
        '3_receipts': multiclass_probs[0, 1].item(),
        '4_receipts': multiclass_probs[0, 2].item(),
        '5_receipts': multiclass_probs[0, 3].item()
    }
    
    # Map multiclass prediction (0-3) back to receipt count (2-5)
    final_prediction = multiclass_prediction + 2
    
    # Calculate overall confidence as a product of confidences at each level
    # Level 1 confidence that it's 1+, Level 2 confidence that it's 2+, Multiclass specific count
    overall_confidence = level1_confidence * level2_confidence * multiclass_confidence
    
    return final_prediction, overall_confidence, confidences
```

## 5. Evaluation Pipeline

```python
def evaluate_hierarchical_model(test_csv, test_dir, model_base_path, output_dir, model_type="vit"):
    """
    Evaluate the hierarchical model on test data.
    """
    # Load test data
    df_test = pd.read_csv(test_csv)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Initialize results lists
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_detailed_confidences = []
    
    # Process each image
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating"):
        image_path = os.path.join(test_dir, row['filename'])
        true_count = row['receipt_count']
        
        # Get prediction using hierarchical model
        predicted_count, confidence, detailed_confidences = hierarchical_predict(
            image_path, 
            model_base_path, 
            model_type=model_type
        )
        
        all_predictions.append(predicted_count)
        all_targets.append(true_count)
        all_confidences.append(confidence)
        all_detailed_confidences.append(detailed_confidences)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    # Per-class metrics
    class_report = classification_report(all_targets, all_predictions, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'filename': df_test['filename'],
        'actual': all_targets,
        'predicted': all_predictions,
        'confidence': all_confidences
    })
    results_df.to_csv(os.path.join(output_dir, 'hierarchical_results.csv'), index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(6), yticklabels=range(6))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Calculate hierarchical performance
    # Level 1 performance (0 vs 1+)
    level1_targets = [0 if t == 0 else 1 for t in all_targets]
    level1_preds = [0 if p == 0 else 1 for p in all_predictions]
    level1_accuracy = accuracy_score(level1_targets, level1_preds)
    level1_balanced_accuracy = balanced_accuracy_score(level1_targets, level1_preds)
    
    # Level 2 performance (1 vs 2+)
    # Filter to only samples with 1+ receipts
    level2_indices = [i for i, t in enumerate(all_targets) if t > 0]
    level2_targets = [1 if all_targets[i] == 1 else 0 for i in level2_indices]
    level2_preds = [1 if all_predictions[i] == 1 else 0 for i in level2_indices]
    level2_accuracy = accuracy_score(level2_targets, level2_preds)
    level2_balanced_accuracy = balanced_accuracy_score(level2_targets, level2_preds)
    
    # Multiclass performance (2-5 receipts)
    # Filter to only samples with 2+ receipts
    multiclass_indices = [i for i, t in enumerate(all_targets) if t > 1]
    multiclass_targets = [all_targets[i] for i in multiclass_indices]
    multiclass_preds = [all_predictions[i] for i in multiclass_indices]
    multiclass_accuracy = accuracy_score(multiclass_targets, multiclass_preds)
    multiclass_balanced_accuracy = balanced_accuracy_score(multiclass_targets, multiclass_preds)
    
    # Print hierarchical performance
    print("\nHierarchical Performance:")
    print(f"Level 1 (0 vs 1+): Accuracy={level1_accuracy:.4f}, Balanced Accuracy={level1_balanced_accuracy:.4f}")
    print(f"Level 2 (1 vs 2+): Accuracy={level2_accuracy:.4f}, Balanced Accuracy={level2_balanced_accuracy:.4f}")
    print(f"Multiclass (2-5): Accuracy={multiclass_accuracy:.4f}, Balanced Accuracy={multiclass_balanced_accuracy:.4f}")
    
    hierarchical_metrics = {
        'overall': {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_macro': f1,
        },
        'level1': {
            'accuracy': level1_accuracy,
            'balanced_accuracy': level1_balanced_accuracy
        },
        'level2': {
            'accuracy': level2_accuracy,
            'balanced_accuracy': level2_balanced_accuracy
        },
        'multiclass': {
            'accuracy': multiclass_accuracy,
            'balanced_accuracy': multiclass_balanced_accuracy
        },
        'class_report': class_report,
        'confusion_matrix': cm,
        'results_csv': os.path.join(output_dir, 'hierarchical_results.csv')
    }
    
    return hierarchical_metrics
```

## 6. Advantages and Considerations

### Advantages
- Each level handles a more balanced classification task
- Level 1 can focus on the crucial distinction between no receipts and having receipts
- Level 2 can specialize in the difficult distinction between single and multiple receipts
- Final multiclass model only needs to distinguish between multiple receipt counts
- Less complex than a 3-level approach but still addresses key imbalance issues

### Considerations
- Need to train and manage three separate models
- Inference requires sequential model execution
- Errors at higher levels propagate to lower levels
- Training data for the multiclass stage is reduced

### Possible Optimizations
- Apply stronger augmentation for minority classes in the multiclass model
- Use model distillation to create smaller, faster models for early levels
- Apply feature sharing between levels to reduce computation
- Ensemble methods at each level for higher accuracy
- Custom loss functions at each level to emphasize problematic cases