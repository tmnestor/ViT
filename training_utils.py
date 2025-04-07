import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os
from config import get_config
from device_utils import get_device, move_to_device


class ModelCheckpoint:
    """
    Model checkpoint utility that saves models based on multiple metrics.
    Supports saving best models based on different metrics simultaneously.
    """
    def __init__(self, output_dir="models", metrics=None, mode="max", prefix="", verbose=True):
        """
        Initialize the model checkpoint utility.
        
        Args:
            output_dir: Directory to save model checkpoints
            metrics: List of metrics to monitor for improvement (e.g., ["accuracy", "balanced_accuracy", "f1_score"])
            mode: 'max' if higher is better, 'min' if lower is better
            prefix: Prefix for saved model filenames
            verbose: Whether to print checkpoint messages
        """
        self.output_dir = output_dir
        self.metrics = metrics or ["balanced_accuracy"]  # Default to balanced accuracy
        self.mode = mode
        self.prefix = prefix
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize best values based on mode
        self.best_values = {}
        for metric in self.metrics:
            self.best_values[metric] = float('-inf') if mode == "max" else float('inf')
        
        # Track if any improvement was made
        self.improved = False
    
    def check_improvement(self, metrics_dict, model, model_type="model"):
        """
        Check if any monitored metric has improved and save the model if so.
        
        Args:
            metrics_dict: Dictionary of metrics (key: metric name, value: metric value)
            model: PyTorch model to save
            model_type: Type of model (e.g., "vit", "swin") for naming
            
        Returns:
            bool: Whether any metric improved
        """
        from model_factory import ModelFactory
        
        self.improved = False
        
        # We only care about f1_macro for saving models
        metric = "f1_macro"
        if metric not in metrics_dict:
            if self.verbose:
                print(f"Warning: F1 macro metric not in provided metrics dictionary. Skipping.")
            return False
        
        current_value = metrics_dict[metric]
        
        # Check if the current value is better than the best value
        if ((self.mode == "max" and current_value > self.best_values[metric]) or
            (self.mode == "min" and current_value < self.best_values[metric])):
            
            # Update best value
            self.best_values[metric] = current_value
            
            # Save model
            save_path = os.path.join(
                self.output_dir, 
                f"{self.prefix}receipt_counter_{model_type}_best.pth"
            )
            ModelFactory.save_model(model, save_path)
            
            if self.verbose:
                print(f"Saved new best model with F1 Macro: {current_value:.4f}")
            
            self.improved = True
        
        return self.improved


class EarlyStopping:
    """
    Early stopping utility that monitors a specified metric and stops training
    if no improvement is seen for a specified number of epochs.
    """
    def __init__(self, patience=5, min_delta=0.001, mode="max", verbose=True):
        """
        Initialize the early stopping utility.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored metric to qualify as improvement
            mode: 'max' if higher is better, 'min' if lower is better
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.should_stop = False
    
    def check_improvement(self, current_value):
        """
        Check if the current value is an improvement over the best value.
        
        Args:
            current_value: Current value of the monitored metric
            
        Returns:
            bool: True if stopping criterion is met, False otherwise
        """
        if ((self.mode == "max" and current_value > self.best_value + self.min_delta) or
            (self.mode == "min" and current_value < self.best_value - self.min_delta)):
            
            # Reset counter if improvement is seen
            self.counter = 0
            self.best_value = current_value
            return False
        else:
            # Increment counter if no improvement is seen
            self.counter += 1
            
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs. "
                      f"Best value: {self.best_value:.4f}, Current value: {current_value:.4f}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training after {self.counter} epochs with no improvement")
                return True
            
            return False


def validate(model, dataloader, criterion, device):
    """
    Validate a model on a validation dataset. Supports both binary and multiclass classification.
    
    Args:
        model: PyTorch model to validate
        dataloader: PyTorch DataLoader with validation data
        criterion: Loss function
        device: Device to run validation on (e.g., "cuda", "cpu", "mps")
        
    Returns:
        dict: Dictionary containing various metrics (loss, accuracy, etc.)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Get number of classes from config
    config = get_config()
    num_classes = len(config.class_distribution)
    
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            # Ensure correct dtypes for MPS compatibility
            images = images.to(device=device, dtype=torch.float32)
            targets = targets.to(device)
            
            # Forward pass - handle different model output formats
            outputs = model(images)
            
            # Handle different model output formats (HF vs. PyTorch)
            if hasattr(outputs, 'logits'):
                # HuggingFace transformer model output
                logits = outputs.logits
                loss = criterion(logits, targets)
                _, predicted = torch.max(logits, 1)
            else:
                # Standard PyTorch model output
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                
            val_loss += loss.item()

            # Calculate accuracy
            batch_size = targets.size(0)
            total += batch_size
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Class-wise accuracy
            for i in range(batch_size):
                label = targets[i].item()
                if label not in class_total:
                    # Initialize if we encounter a new label not in our dictionary
                    class_total[label] = 0
                    class_correct[label] = 0
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1

    # Overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Class-balanced accuracy
    class_accuracies = []
    for cls in range(num_classes):
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls]
            class_accuracies.append(cls_acc)
    
    balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0
    
    # Calculate F1 scores
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    
    # Create detailed metrics dictionary
    metrics = {
        'loss': val_loss / len(dataloader),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'predictions': all_preds,
        'targets': all_targets,
        'class_accuracies': class_accuracies,
    }
    
    return metrics


def print_validation_results(metrics, verbose=True):
    """
    Print validation results in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from validate()
        verbose: Whether to print detailed results
    """
    if not verbose:
        return
    
    # Get basic metrics
    loss = metrics['loss']
    accuracy = metrics['accuracy']
    balanced_accuracy = metrics['balanced_accuracy']
    f1_macro = metrics['f1_macro']
    
    print(f"\nValidation Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"F1 Macro: {f1_macro:.2%}")
    
    # Print per-class accuracies if available
    if 'class_accuracies' in metrics and metrics['class_accuracies']:
        print("\nClass Accuracies:")
        for i, acc in enumerate(metrics['class_accuracies']):
            print(f"  Class {i}: {acc:.2%}")


def plot_confusion_matrix(predictions, ground_truth, output_path=None, figsize=(12, 10)):
    """
    Plot a confusion matrix for classification results.
    
    Args:
        predictions: List or array of model predictions
        ground_truth: List or array of ground truth labels
        output_path: Path to save the plot (if None, the plot will be shown but not saved)
        figsize: Figure size (width, height) in inches
        
    Returns:
        tuple: (accuracy, balanced_accuracy)
    """
    plt.figure(figsize=figsize)
    
    # Get number of classes from config
    config = get_config()
    num_classes = len(config.class_distribution)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Use descriptive class labels
    class_labels = ["0", "1", "2+"] if num_classes == 3 else [str(i) for i in range(num_classes)]
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Calculate metrics
    correct = np.sum(np.diag(cm))
    total = np.sum(cm)
    accuracy = correct / total if total > 0 else 0
    
    # Class accuracies
    class_accuracies = []
    for i in range(cm.shape[0]): # Use cm.shape[0] to ensure we only iterate over available classes
        total_class = np.sum(cm[i, :])
        if total_class > 0:
            class_accuracies.append(cm[i, i] / total_class)
    balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0
    
    # Add summary statistics
    stats_text = f"Overall Accuracy: {accuracy:.2%}\nBalanced Accuracy: {balanced_accuracy:.2%}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=12)
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return accuracy, balanced_accuracy


def plot_training_curves(history, output_path=None, figsize=(15, 8)):
    """
    Plot training and validation loss, accuracy, and other metrics.
    
    Args:
        history: Dictionary with training history
                (should include train_loss, val_loss, val_acc, val_balanced_acc, val_f1_macro)
        output_path: Path to save the plot (if None, the plot will be shown but not saved)
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history["val_acc"], label="Accuracy")
    plt.plot(history["val_balanced_acc"], label="Balanced Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 macro plot
    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1_macro"], label="F1 Macro", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Macro Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path)
        plt.close()


def plot_evaluation_metrics(metrics, output_dir="evaluation"):
    """
    Create and save evaluation plots for a classification model.
    
    Args:
        metrics: Dictionary with evaluation metrics (should include predictions, targets)
        output_dir: Directory to save the plots
        
    Returns:
        tuple: (accuracy, balanced_accuracy)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the appropriate predictions (calibrated or uncalibrated)
    if 'calibrated_predictions' in metrics:
        predictions = metrics['calibrated_predictions']
        print("DEBUG: Using calibrated predictions for evaluation plots")
    else:
        predictions = metrics['predictions']
        print("DEBUG: Using uncalibrated predictions for evaluation plots")
    
    ground_truth = metrics['targets']
    
    # Get number of classes from config
    config = get_config()
    num_classes = len(config.class_distribution)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    confusion_mat = confusion_matrix(ground_truth, predictions)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    class_labels = ["0", "1", "2+"] if num_classes == 3 else [f'{i}' for i in range(num_classes)]
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Normalize confusion matrix by row (true label)
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    norm_conf_mx = confusion_mat / row_sums
    
    # Normalized Confusion Matrix
    plt.subplot(2, 2, 2)
    plt.imshow(norm_conf_mx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    plt.xticks(tick_marks, [f'{i}' for i in range(num_classes)])
    plt.yticks(tick_marks, [f'{i}' for i in range(num_classes)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Calculate class accuracies
    class_acc = []
    for i in range(num_classes):
        class_total = np.sum(np.array(ground_truth) == i)
        if class_total > 0:
            class_correct = confusion_mat[i, i]
            class_acc.append(class_correct / class_total)
        else:
            class_acc.append(0.0)
    
    # Class Accuracy Bar Chart
    plt.subplot(2, 2, 3)
    plt.bar(range(len(class_acc)), class_acc)
    plt.title('Per-Class Accuracy')
    class_labels = ["0", "1", "2+"] if num_classes == 3 else [f'{i}' for i in range(num_classes)]
    plt.xticks(range(len(class_acc)), class_labels)
    plt.xlabel('Number of Receipts')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    
    # Class Distribution
    plt.subplot(2, 2, 4)
    class_counts = np.array([np.sum(np.array(ground_truth) == i) for i in range(num_classes)])
    plt.bar(range(len(class_counts)), class_counts)
    plt.title('Class Distribution')
    class_labels = ["0", "1", "2+"] if num_classes == 3 else [f'{i}' for i in range(num_classes)]
    plt.xticks(range(len(class_counts)), class_labels)
    plt.xlabel('Number of Receipts')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=300)
    
    # Get classification report
    class_report = classification_report(ground_truth, predictions, output_dict=True)
    
    # Save error analysis by class count
    error_by_count = []
    for i in range(num_classes):
        mask = np.array(ground_truth) == i
        if np.sum(mask) > 0:
            class_pred = np.array(predictions)[mask]
            class_gt = np.array(ground_truth)[mask]
            error_by_count.append({
                'count': i,
                'samples': np.sum(mask),
                'accuracy': np.mean(class_pred == class_gt),
                'f1_score': class_report[str(i)]['f1-score'],
                'precision': class_report[str(i)]['precision'],
                'recall': class_report[str(i)]['recall']
            })
    
    # Plot error by count
    plt.figure(figsize=(10, 6))
    count_vals = [item['count'] for item in error_by_count]
    plt.bar(count_vals, [item['accuracy'] for item in error_by_count], label='Accuracy')
    plt.plot(count_vals, [item['f1_score'] for item in error_by_count], 'o-', color='red', label='F1 Score')
    plt.title('Performance by Number of Receipts')
    # Use descriptive labels for the 3-class system
    if len(count_vals) == 3:
        plt.xticks(count_vals, ["0", "1", "2+"])
    else:
        plt.xticks(count_vals)
    plt.xlabel('Number of Receipts')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_by_count.png"), dpi=300)
    plt.close()
    
    # Calculate overall metrics
    accuracy = np.mean(np.array(predictions) == np.array(ground_truth))
    balanced_accuracy = np.mean(class_acc)
    
    return accuracy, balanced_accuracy