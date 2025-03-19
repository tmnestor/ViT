import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from vit_counter import ViTReceiptCounter
from receipt_processor import ReceiptProcessor
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


# Custom dataset for receipt counting (as classification)
class ReceiptDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or ReceiptProcessor().transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        # Receipt count as target class (0-5)
        count = int(self.data.iloc[idx, 1])
        return image, torch.tensor(count, dtype=torch.long)


def validate(model, dataloader, criterion, device):
    """Validate the model on a validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(6)}  # 0-5 receipts
    class_total = {i: 0 for i in range(6)}    # 0-5 receipts
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
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
    for cls in range(6):
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls]
            class_accuracies.append(cls_acc)
    
    balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0
    
    # Debug information
    print("\nClassification Report:")
    for cls in range(6):
        if class_total[cls] > 0:
            cls_acc = class_correct[cls] / class_total[cls]
            print(f"  Class {cls}: {class_correct[cls]}/{class_total[cls]} ({cls_acc:.2%})")
    
    # Create confusion matrix
    conf_matrix = np.zeros((6, 6), dtype=int)
    for pred, target in zip(all_preds, all_targets):
        conf_matrix[int(target), int(pred)] += 1
    
    print("\nConfusion Matrix:")
    print("  " + " ".join(f"{i:4d}" for i in range(6)))
    for i in range(6):
        print(f"{i} {' '.join(f'{conf_matrix[i, j]:4d}' for j in range(6))}")

    return val_loss / len(dataloader), accuracy, balanced_accuracy, all_preds, all_targets


def plot_results(predictions, ground_truth, output_path="classification_results.png"):
    """Plot confusion matrix and classification results."""
    plt.figure(figsize=(12, 10))
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['0', '1', '2', '3', '4', '5']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
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
    for i in range(len(classes)):
        total_class = np.sum(cm[i, :])
        if total_class > 0:
            class_accuracies.append(cm[i, i] / total_class)
    balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0
    
    # Add summary statistics
    stats_text = f"Overall Accuracy: {accuracy:.2%}\nBalanced Accuracy: {balanced_accuracy:.2%}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=12)
    
    plt.savefig(output_path)
    
    return accuracy, balanced_accuracy


def train_model(
    train_csv,
    train_dir,
    val_csv=None,
    val_dir=None,
    epochs=15,
    batch_size=16,
    lr=1e-4,
    output_dir="models",
):
    """
    Train the ViT-Base model for receipt counting as a classification task.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using MPS with fallback enabled")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize training dataset and loader
    train_dataset = ReceiptDataset(train_csv, train_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Initialize validation dataset and loader if provided
    if val_csv and val_dir:
        val_dataset = ReceiptDataset(val_csv, val_dir)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        print(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
    else:
        # Create a validation split from training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        print(
            f"Split {len(train_dataset)} samples into {train_size} training and {val_size} validation"
        )

    # Initialize model as classification model
    model = ViTReceiptCounter(pretrained=True, num_classes=6).to(device)
    print("Initialized ViT-Base model for receipt counting (classification)")

    # Loss and optimizer with more robust learning rate control
    criterion = nn.CrossEntropyLoss()  # Standard classification loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Use ReduceLROnPlateau scheduler to reduce LR when validation metrics plateau
    # This helps prevent erratic bouncing around the optimum
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Monitor balanced accuracy which we want to maximize
        factor=0.5,          # Multiply LR by this factor on plateau
        patience=2,          # Number of epochs with no improvement before reducing LR
        verbose=True,        # Print message when LR is reduced
        min_lr=1e-6          # Don't reduce LR below this value
    )

    # Training metrics
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": []}

    # For early stopping
    patience = 5
    patience_counter = 0
    best_balanced_acc = 0

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping to prevent large updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Learning rate scheduler will be updated after validation

        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss, val_acc, val_balanced_acc, predictions, ground_truth = validate(
            model, val_loader, criterion, device
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Balanced Accuracy: {val_balanced_acc:.2%}"
        )
        
        # Update learning rate scheduler based on balanced accuracy
        scheduler.step(val_balanced_acc)

        # Save model on improvement
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            model.save(os.path.join(output_dir, "receipt_counter_vit_best.pth"))
            print(f"Saved best model with balanced accuracy: {val_balanced_acc:.2%}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save final model
    model.save(os.path.join(output_dir, "receipt_counter_vit_final.pth"))

    # Generate validation plots
    _, _, _, predictions, ground_truth = validate(model, val_loader, criterion, device)
    accuracy, balanced_accuracy = plot_results(
        predictions,
        ground_truth,
        output_path=os.path.join(output_dir, "vit_classification_results.png"),
    )

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "vit_classification_history.csv"), index=False
    )

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Accuracy")
    plt.plot(history["val_balanced_acc"], label="Balanced Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vit_classification_curves.png"))

    print("\nFinal Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")

    print(f"\nTraining complete! Models saved to {output_dir}/")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a ViT model for receipt counting (classification)"
    )
    parser.add_argument(
        "--train_csv",
        default="receipt_dataset/train.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--train_dir",
        default="receipt_dataset/train",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--val_csv",
        default="receipt_dataset/val.csv",
        help="Path to validation CSV file",
    )
    parser.add_argument(
        "--val_dir",
        default="receipt_dataset/val",
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Directory to save trained model and results",
    )

    args = parser.parse_args()

    # Validate that files exist
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV file not found: {args.train_csv}")
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Training directory not found: {args.train_dir}")

    # Optional validation files
    val_csv = args.val_csv if os.path.exists(args.val_csv) else None
    val_dir = args.val_dir if os.path.exists(args.val_dir) else None

    train_model(
        args.train_csv,
        args.train_dir,
        val_csv,
        val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()