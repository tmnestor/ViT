import torch
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from vit_counter import ViTReceiptCounter
from train_vit_counter import ReceiptDataset

def validate(model, dataloader, criterion, device):
    """Validate the model on a validation set."""
    model.eval()
    val_loss = 0.0
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for images, counts in dataloader:
            images, counts = images.to(device), counts.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, counts)
            val_loss += loss.item()
            
            # Store predictions and ground truth
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(counts.cpu().numpy())
    
    # Calculate MAE and other metrics
    mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))
    exact_match = np.mean(np.round(predictions) == np.round(ground_truth))
    
    return val_loss / len(dataloader), mae, exact_match, predictions, ground_truth

def evaluate_model(model_path, test_csv, test_dir, batch_size=16, output_dir="evaluation_vit"):
    """
    Evaluate a trained ViT receipt counter model on test data.
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to CSV file containing test data
        test_dir: Directory containing test images
        batch_size: Batch size for evaluation
        output_dir: Directory to save evaluation results
    """
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
    print(f"Loading ViT model from {model_path}...")
    model = ViTReceiptCounter.load(model_path).to(device)
    model.eval()
    
    # Initialize dataset and loader
    test_dataset = ReceiptDataset(test_csv, test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Evaluation
    print("Evaluating ViT model...")
    criterion = torch.nn.MSELoss()
    _, mae, exact_match, predictions, ground_truth = validate(model, test_loader, criterion, device)
    
    # Calculate additional metrics
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truth, predictions)
    
    # Print results
    print("\nViT Evaluation Results:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Exact Match Rate: {exact_match:.2%}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'filename': test_dataset.data.iloc[:, 0],
        'actual': ground_truth,
        'predicted': predictions,
        'rounded_pred': np.round(predictions),
        'error': np.array(predictions) - np.array(ground_truth),
        'abs_error': np.abs(np.array(predictions) - np.array(ground_truth))
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Prediction vs ground truth scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(ground_truth, predictions, alpha=0.5)
    max_val = max(max(predictions), max(ground_truth))
    min_val = min(min(predictions), min(ground_truth))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('ViT: Predicted vs Actual Receipt Counts')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 2, 2)
    errors = np.array(predictions) - np.array(ground_truth)
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Prediction error by actual count
    plt.subplot(2, 2, 3)
    plt.scatter(ground_truth, np.abs(errors), alpha=0.5)
    plt.title('Absolute Error vs Actual Count')
    plt.xlabel('Actual Count')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix-like visualization for count accuracy
    plt.subplot(2, 2, 4)
    max_count = int(max(max(np.round(predictions)), max(ground_truth)))
    min_count = int(min(min(np.round(predictions)), min(ground_truth)))
    count_range = range(min_count, max_count + 1)
    
    conf_matrix = np.zeros((len(count_range), len(count_range)))
    for true, pred in zip(ground_truth, predictions):
        true_idx = int(round(true)) - min_count
        pred_idx = int(round(pred)) - min_count
        # Ensure indices are within bounds
        if 0 <= true_idx < len(count_range) and 0 <= pred_idx < len(count_range):
            conf_matrix[true_idx, pred_idx] += 1
    
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar(label='Count')
    plt.title('Count Prediction Confusion Matrix')
    plt.xlabel('Predicted Count')
    plt.ylabel('Actual Count')
    
    # Add count labels
    tick_positions = np.arange(len(count_range))
    plt.xticks(tick_positions, count_range)
    plt.yticks(tick_positions, count_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"))
    
    # Error analysis by count
    error_by_count = results_df.groupby('actual').agg({
        'abs_error': ['mean', 'std', 'count'],
        'error': 'mean'
    })
    error_by_count.columns = ['mae', 'std', 'count', 'bias']
    error_by_count.to_csv(os.path.join(output_dir, "error_by_count.csv"))
    
    # Plot error by count
    plt.figure(figsize=(10, 6))
    plt.bar(error_by_count.index, error_by_count['mae'], alpha=0.7)
    plt.errorbar(error_by_count.index, error_by_count['mae'], 
                 yerr=error_by_count['std'], fmt='o', color='r')
    plt.title('ViT: Mean Absolute Error by Receipt Count')
    plt.xlabel('Actual Receipt Count')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_by_count.png"))
    
    print(f"\nEvaluation complete! Results saved to {output_dir}/")
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact_match': exact_match,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT receipt counter model")
    parser.add_argument("--model", required=True, 
                       help="Path to the trained model")
    parser.add_argument("--test_csv", required=True,
                       help="Path to CSV file containing test data")
    parser.add_argument("--test_dir", required=True,
                       help="Directory containing test images")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--output_dir", default="evaluation_vit",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model, args.test_csv, args.test_dir,
        batch_size=args.batch_size, output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()