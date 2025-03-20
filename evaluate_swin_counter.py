import torch
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Import model creation functions
from transformer_swin import create_swin_transformer, load_swin_model
from config import get_config
from simple_receipt_counter import create_simple_hf_receipt_counter

# Standalone dataset class to avoid import from train_swin_classification
class ReceiptCollageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, binary=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.root_dir = os.path.dirname(self.img_dir)
        self.binary = binary  # Flag for binary classification
        
        # No transform dependency - using simple resize and normalization
        self.image_size = 224  # Standard size for ViT and Swin models
        # ImageNet mean and std for normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Print the first few file names in the dataset
        print(f"First few files in dataset: {self.data.iloc[:5, 0].tolist()}")
        print(f"Checking for image files in: {self.img_dir} and parent dir")
        if binary:
            print("Using binary classification mode (0 vs 1+ receipts)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        
        # Try multiple potential locations for the image
        potential_paths = [
            os.path.join(self.img_dir, filename),                # Primary location
            os.path.join(self.root_dir, 'train', filename),      # Alternative in train dir
            os.path.join(self.root_dir, 'val', filename),        # Alternative in val dir
            os.path.join(self.root_dir, filename)                # Alternative in root dir
        ]
        
        # Try each potential path
        image = None
        for path in potential_paths:
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                break
        
        if image is None:
            # Fallback to using a blank image rather than crashing
            print(f"Warning: Could not find image {filename} in any potential location.")
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Resize image to standard size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Manual normalization using ImageNet stats
        image_np = (image_np - self.mean) / self.std
        
        # Convert to tensor in CxHxW format
        image_tensor = torch.tensor(image_np).permute(2, 0, 1)

        # Receipt count as target class
        count = int(self.data.iloc[idx, 1])
        
        if self.binary:
            # Convert to binary classification (0 vs 1+ receipts)
            binary_label = 1 if count > 0 else 0
            return image_tensor, torch.tensor(binary_label, dtype=torch.long)
        else:
            # Original multiclass classification (0-5)
            return image_tensor, torch.tensor(count, dtype=torch.long)

# Standalone validate function to avoid import from train_swin_classification
def validate(model, dataloader, criterion, device):
    """Validate the model on a validation set."""
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
            
            # Hugging Face models return an output object with logits
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
    
    # Debug information
    print("\nClassification Report:")
    for cls in range(num_classes):
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls]
            print(f"  Class {cls}: {class_correct[cls]}/{class_total[cls]} ({cls_acc:.2%})")
    
    return val_loss / len(dataloader), accuracy, balanced_accuracy, all_preds, all_targets

def evaluate_model(model_path, test_csv, test_dir, batch_size=16, output_dir="evaluation", mode="classification", config_path=None, binary=False):
    """
    Evaluate a trained receipt counter model on test data.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to CSV file containing test data
        test_dir: Directory containing test images
        batch_size: Batch size for evaluation
        output_dir: Directory to save evaluation results
        mode: Evaluation mode ('classification' or 'regression')
        config_path: Path to configuration JSON file (optional)
        binary: If True, use binary classification (0 vs 1+ receipts)
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using MPS with fallback enabled")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading Swin model from {model_path}...")
    
    try:
        # First attempt: Try to load using the specific loader function
        try:
            # Try to load the saved weights with our loader
            model = load_swin_model(model_path)
            print("Successfully loaded weights into Swin model!")
        except Exception as e:
            print(f"Could not load directly with load_swin_model: {e}")
            
            # Try creating a new model and loading the state dict directly
            print("Creating new Hugging Face Swin model and loading weights...")
            # Load the saved state dict
            saved_state_dict = torch.load(model_path)
            print(f"Loaded state dict with keys: {list(saved_state_dict.keys())[:5]}...")
            
            model = create_swin_transformer(pretrained=False, verbose=False)
            try:
                # First try with strict=True
                model.load_state_dict(saved_state_dict)
                print("Loaded weights with strict=True")
            except Exception as e:
                print(f"Strict loading failed: {e}")
                print("Attempting to load with strict=False as fallback...")
                model.load_state_dict(saved_state_dict, strict=False)
                print("Loaded weights with strict=False, but this may indicate a model architecture mismatch")
            
        # Move model to device
        model = model.to(device)
    except Exception as e:
        print(f"Error during model loading: {e}")
        print("Falling back to using pretrained HF model without custom weights")
        
        # Initialize a basic model as fallback
        # Disable warning messages from HuggingFace
        import transformers
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        
        # Get number of classes from config
        config = get_config()
        num_classes = len(config.class_distribution)
        model = create_simple_hf_receipt_counter(num_classes=num_classes, model_type="swin")
        
        # Restore verbosity
        transformers.logging.set_verbosity(prev_verbosity)
        if model is None:
            print("Error: Failed to create fallback model")
            return None
            
        model = model.to(device)
    
    model.eval()
    
    # Initialize dataset and loader
    test_dataset = ReceiptCollageDataset(test_csv, test_dir, binary=binary)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Evaluation
    print("Evaluating Swin model...")
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, accuracy, balanced_accuracy, predictions, ground_truth = validate(model, test_loader, criterion, device)
    
    # Calculate class accuracies and confusion matrix (since they're not returned by validate)
    num_classes = len(get_config().class_distribution)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(predictions, ground_truth):
        confusion_mat[int(target), int(pred)] += 1
        
    class_acc = []
    for i in range(num_classes):
        class_total = np.sum(np.array(ground_truth) == i)
        if class_total > 0:
            class_correct = confusion_mat[i, i]
            class_acc.append(class_correct / class_total)
        else:
            class_acc.append(0.0)
    
    # Calculate additional metrics
    class_report = classification_report(ground_truth, predictions, output_dict=True)
    f1_macro = class_report['macro avg']['f1-score']
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Loss: {val_loss:.3f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"F1-Macro: {f1_macro:.2%}")
    print("\nClass Accuracies:")
    for i, acc in enumerate(class_acc):
        print(f"  Class {i} (receipts): {acc:.2%}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'filename': test_dataset.data.iloc[:, 0],
        'actual': ground_truth,
        'predicted': predictions,
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f'{i}' for i in range(num_classes)])
    plt.yticks(tick_marks, [f'{i}' for i in range(num_classes)])
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
    
    # Class Accuracy Bar Chart
    plt.subplot(2, 2, 3)
    plt.bar(range(len(class_acc)), class_acc)
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_acc)), [f'{i}' for i in range(num_classes)])
    plt.xlabel('Number of Receipts')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    
    # Class Distribution
    plt.subplot(2, 2, 4)
    class_counts = np.array([np.sum(np.array(ground_truth) == i) for i in range(num_classes)])
    plt.bar(range(len(class_counts)), class_counts)
    plt.title('Class Distribution')
    plt.xticks(range(len(class_counts)), [f'{i}' for i in range(num_classes)])
    plt.xlabel('Number of Receipts')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=300)
    
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
    
    error_df = pd.DataFrame(error_by_count)
    error_df.to_csv(os.path.join(output_dir, "error_by_count.csv"), index=False)
    
    # Plot error by count
    plt.figure(figsize=(10, 6))
    count_vals = error_df['count'].values
    plt.bar(count_vals, error_df['accuracy'], label='Accuracy')
    plt.plot(count_vals, error_df['f1_score'], 'o-', color='red', label='F1 Score')
    plt.title('Performance by Number of Receipts')
    plt.xticks(count_vals)
    plt.xlabel('Number of Receipts')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_by_count.png"), dpi=300)
    plt.close()
    
    print(f"\nEvaluation complete! Results saved to {output_dir}/")
    return {
        'loss': val_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'class_accuracies': class_acc,
        'confusion_matrix': confusion_mat
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Swin-Tiny receipt counter model")
    parser.add_argument("--model", required=True, 
                       help="Path to the trained model")
    parser.add_argument("--test_csv", required=True,
                       help="Path to CSV file containing test data")
    parser.add_argument("--test_dir", required=True,
                       help="Directory containing test images")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--model_variant", default="best",
                       choices=["best", "best_bacc", "best_f1", "final"],
                       help="Model variant to evaluate (default: best)")
    parser.add_argument("--output_dir", default="evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--config", 
                       help="Path to configuration JSON file")
    parser.add_argument("--binary", action="store_true",
                       help="Evaluate as binary classification (multiple receipts or not)")
    
    args = parser.parse_args()
    
    # If a model directory is provided instead of a specific model file,
    # construct the path using the model_variant argument
    model_path = args.model
    if os.path.isdir(model_path):
        variant_map = {
            "best": "receipt_counter_swin_best.pth",
            "best_bacc": "receipt_counter_swin_best_bacc.pth",
            "best_f1": "receipt_counter_swin_best_f1.pth",
            "final": "receipt_counter_swin_final.pth"
        }
        model_filename = variant_map.get(args.model_variant, "receipt_counter_swin_best.pth")
        model_path = os.path.join(model_path, model_filename)
        print(f"Using model variant: {args.model_variant} at path: {model_path}")
    
    evaluate_model(
        model_path, args.test_csv, args.test_dir,
        batch_size=args.batch_size, output_dir=args.output_dir,
        config_path=args.config, binary=args.binary
    )

if __name__ == "__main__":
    main()