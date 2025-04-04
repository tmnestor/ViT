"""
Unified evaluation module for transformer models.
This module centralizes evaluation for both ViT and Swin models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from model_factory import ModelFactory
from config import get_config
from datasets import ReceiptDataset
from training_utils import validate, plot_evaluation_metrics
from device_utils import get_device


def evaluate_model(
    model_path, 
    test_csv, 
    test_dir, 
    batch_size=16, 
    output_dir="evaluation", 
    config_path=None, 
    binary=False,
    apply_calibration=True
):
    """
    Evaluate a trained transformer model on test data.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to CSV file containing test data
        test_dir: Directory containing test images
        batch_size: Batch size for evaluation
        output_dir: Directory to save evaluation results
        model_type: Type of model to evaluate ("vit" or "swin")
        config_path: Path to configuration JSON file (optional)
        binary: If True, use binary classification (0 vs 1+ receipts)
        apply_calibration: If True, apply Bayesian calibration to predictions (default: True)
                          This makes evaluation consistent with individual_image_tester.py
        
    Returns:
        dict: Dictionary containing evaluation metrics and results
        
    Note: This function specifically loads the model from the path provided in model_path, 
    not the most recently trained model.
    """
    
    # Load configuration
    config = get_config()
    if config_path:
        if os.path.exists(config_path):
            config.load_from_file(config_path, silent=False)  # Explicitly show this load
        else:
            print(f"Warning: Configuration file not found: {config_path}")
            print("Using default configuration")
    
    # Set binary mode if specified
    if binary:
        config.set_binary_mode(True)
        print("Using binary classification mode (0 vs 1+ receipts)")
    else:
        config.set_binary_mode(False)
    
    print(f"Using class distribution: {config.class_distribution}")
    print(f"Using calibration factors: {config.calibration_factors}")
    
    # Load configuration with calibration factors
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the best available device
    device = get_device()
    
    # Load model as SwinV2
    print(f"Loading model from {model_path} as SwinV2...")
    
    try:
        # Use the ModelFactory.load_model which now only supports SwinV2
        model = ModelFactory.load_model(
            model_path, 
            strict=False,  # Try non-strict loading if strict fails
            mode="eval"
        )
        
        print(f"Successfully loaded weights into SwinV2 model!")
        
        # SwinV2 is the only supported model type
        model_type = "swinv2"
            
        # Move model to device
        model = model.to(device)
    except Exception as e:
        print(f"Error during model loading: {e}")
        print("Failed to load model. Please ensure the model file exists and is compatible.")
        print("Try using the correct evaluator script for your model type:")
        print("  - evaluate_vit_counter.py for ViT models")
        print("  - evaluate_swin_counter.py for Swin models")
        print("  - evaluate_swinv2_classifier.py for SwinV2 models")
        return None
    
    model.eval()
    
    # Initialize dataset and loader
    test_dataset = ReceiptDataset(test_csv, test_dir, binary=binary)
    
    # Initialize test dataset
    
    # Get num_workers from config
    num_workers = config.get_model_param("num_workers", 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Evaluation
    print("Evaluating SwinV2 model...")
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use the unified validation function that returns a metrics dictionary
    print("Running evaluation using the SwinV2 architecture...")
    metrics = validate(model, test_loader, criterion, device)
    
    # If calibration requested, get calibrated predictions
    if apply_calibration and not binary:
        print("Applying Bayesian calibration to predictions...")
        # Get raw logits from the model
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                # Get model outputs
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                all_logits.append(logits)
                all_targets.append(targets)
        
        # Concatenate all logits and targets
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(all_logits, dim=1)
        
        # Apply calibration
        class_prior = config.get_class_prior_tensor(device)
        calibration_factors = config.get_calibration_tensor(device)
        
        # Apply calibration to each sample
        calibrated_probs = probs * calibration_factors.unsqueeze(0) * class_prior.unsqueeze(0)
        
        # Normalize
        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)
        
        # Get calibrated predictions
        calibrated_predictions = torch.argmax(calibrated_probs, dim=1).cpu().numpy()
        
        # Calculate calibrated metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        calib_accuracy = accuracy_score(all_targets.cpu().numpy(), calibrated_predictions)
        calib_balanced_accuracy = balanced_accuracy_score(all_targets.cpu().numpy(), calibrated_predictions)
        calib_f1_macro = f1_score(all_targets.cpu().numpy(), calibrated_predictions, average='macro')
        
        # Add calibrated metrics to the results
        metrics['calibrated_accuracy'] = calib_accuracy
        metrics['calibrated_balanced_accuracy'] = calib_balanced_accuracy
        metrics['calibrated_f1_macro'] = calib_f1_macro
        metrics['calibrated_predictions'] = calibrated_predictions.tolist()
        
        # Update the predictions to use calibrated ones
        predictions = calibrated_predictions
        print(f"Calibrated metrics - Accuracy: {calib_accuracy:.2%}, Balanced Accuracy: {calib_balanced_accuracy:.2%}, F1 Macro: {calib_f1_macro:.2%}")
    else:
        # Use uncalibrated metrics
        predictions = metrics['predictions']
    
    # Extract metrics for reporting
    val_loss = metrics['loss']
    accuracy = metrics['calibrated_accuracy'] if apply_calibration else metrics['accuracy']
    balanced_accuracy = metrics['calibrated_balanced_accuracy'] if apply_calibration else metrics['balanced_accuracy']
    f1_macro = metrics['calibrated_f1_macro'] if apply_calibration else metrics['f1_macro']
    ground_truth = metrics['targets']
    
    # Print results
    print(f"\n{model_type.upper()} Evaluation Results:")
    print(f"Loss: {val_loss:.3f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"F1-Macro: {f1_macro:.2%}")
    
    if 'class_accuracies' in metrics:
        print("\nClass Accuracies:")
        class_labels = ["0 receipts", "1 receipt", "2+ receipts"]
        for i, acc in enumerate(metrics['class_accuracies']):
            print(f"  Class {i} ({class_labels[i]}): {acc:.2%}")
    
    # Save predictions using the appropriate predictions based on calibration setting
    results_df = pd.DataFrame({
        'filename': test_dataset.data.iloc[:, 0],
        'actual': ground_truth,
        'predicted': metrics['calibrated_predictions'] if (apply_calibration and 'calibrated_predictions' in metrics) else metrics['predictions'],
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Use the unified plotting function to generate all evaluation plots
    metrics_accuracy, metrics_balanced_accuracy = plot_evaluation_metrics(metrics, output_dir=output_dir)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}/")
    
    # Return the metrics dictionary as is, with additional information
    metrics['output_dir'] = output_dir
    metrics['results_file'] = os.path.join(output_dir, "evaluation_results.csv")
    
    return metrics