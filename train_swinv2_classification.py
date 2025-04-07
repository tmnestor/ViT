import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os  # Keep for environment variables and some legacy functionality
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_factory import ModelFactory
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from config import get_config
from training_utils import validate, plot_confusion_matrix, plot_training_curves, ModelCheckpoint, EarlyStopping
from device_utils import get_device
from reproducibility import set_seed, get_reproducibility_info


# Import dataset classes from the unified module
from datasets import ReceiptDataset, create_data_loaders


# Using unified validation and plotting functions from training_utils.py


def train_model(
    train_csv,
    train_dir,
    val_csv=None,
    val_dir=None,
    epochs=None,
    batch_size=None,
    lr=None,
    output_dir=None,
    binary=False,
    augment=None,
    resume_checkpoint=None,
    model_type=None,
    offline=False,
    pretrained_model_dir=None,
):
    """
    Train the SwinV2 Transformer model for receipt counting as a classification task.
    
    Args:
        train_csv: Path to training CSV file
        train_dir: Directory containing training images
        val_csv: Path to validation CSV file (optional)
        val_dir: Directory containing validation images (optional)
        epochs: Number of training epochs (default from config)
        batch_size: Batch size for training (default from config)
        lr: Learning rate (default from config)
        output_dir: Directory to save trained model and results (default from config)
        binary: If True, use binary classification (0 vs 1+ receipts)
        augment: If True, apply data augmentation during training (default from config)
        resume_checkpoint: Path to checkpoint to resume training from (optional)
        model_type: Type of SwinV2 model to use ("swinv2" or "swinv2-large") (default from config)
        offline: If True, use locally downloaded model weights without online access
        pretrained_model_dir: Directory containing pre-downloaded model weights (used with offline=True)
        
    Returns:
        The best model from training, NOT the final model from the last epoch
    """
    # Get configuration singleton
    config = get_config()
    
    # Get parameters from config or use provided values
    if model_type is None:
        model_type = config.get_model_param("model_type", "swinv2")
    
    if lr is None:
        lr = config.get_model_param("learning_rate", 5e-5)
    
    if batch_size is None:
        batch_size = config.get_model_param("batch_size", 8)
    
    if epochs is None:
        epochs = config.get_model_param("epochs", 30)
    
    if output_dir is None:
        output_dir = config.get_model_param("output_dir", "models")
    
    if augment is None:
        augment = config.get_model_param("data_augmentation", True)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure for binary classification if requested
    if binary:
        config.set_binary_mode(True)
        print("Using binary classification (multiple receipts or not)")
    else:
        config.set_binary_mode(False)
        if len(config.class_distribution) == 3:
            print("Using simplified 3-class classification (0, 1, 2+ receipts)")
        else:
            print(f"Using multi-class classification (0-{len(config.class_distribution)-1} receipts)")

    # Get additional parameters from config
    num_workers = config.get_model_param("num_workers", 4)
    
    # Create data loaders using the unified function
    train_loader, val_loader, num_train_samples, num_val_samples = create_data_loaders(
        train_csv=Path(train_csv),
        train_dir=Path(train_dir),
        val_csv=Path(val_csv) if val_csv else None,
        val_dir=Path(val_dir) if val_dir else None,
        batch_size=batch_size,
        augment_train=augment,
        binary=binary
    )

    # Set random seed for reproducibility
    seed_info = set_seed()
    print(f"Using random seed: {seed_info['seed']}, deterministic mode: {seed_info['deterministic']}")
    
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model or load from checkpoint
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        print(f"Loading model checkpoint from {checkpoint_path}")
        model = ModelFactory.load_model(checkpoint_path, model_type=model_type).to(device)
        print(f"Resumed {model_type} Transformer model from checkpoint")
    else:
        if offline:
            print(f"Using offline mode with pre-downloaded model weights")
            
            # Check directory name for hints about model type to avoid mismatches
            if pretrained_model_dir:
                # Auto-detect model type from directory name
                dir_path = Path(pretrained_model_dir)
                dir_name = dir_path.name.lower()
                
                # If directory name contains model type hint that doesn't match requested type
                if 'large' in dir_name and model_type != 'swinv2-large':
                    print(f"WARNING: Directory name suggests a SwinV2-Large model, but requested type is {model_type}")
                    print(f"Auto-switching to swinv2-large model type to match the pre-downloaded weights")
                    model_type = 'swinv2-large'
                elif 'tiny' in dir_name and model_type == 'swinv2-large':
                    print(f"WARNING: Directory name suggests a SwinV2-Tiny model, but requested type is {model_type}")
                    print(f"Auto-switching to swinv2 (tiny) model type to match the pre-downloaded weights")
                    model_type = 'swinv2'
                
                print(f"Loading from specified directory: {pretrained_model_dir} as model type: {model_type}")
                model = ModelFactory.create_transformer(
                    model_type=model_type, 
                    pretrained=True, 
                    offline=True,
                    pretrained_model_dir=pretrained_model_dir
                ).to(device)
            else:
                print(f"Loading from default cache location (offline mode)")
                model = ModelFactory.create_transformer(
                    model_type=model_type, 
                    pretrained=True, 
                    offline=True
                ).to(device)
        else:
            model = ModelFactory.create_transformer(model_type=model_type, pretrained=True).to(device)
            print(f"Initialized new {model_type} Transformer model using Hugging Face transformers")

    # Loss and optimizer with more robust learning rate control
    # Get class weights from configuration system
    print(f"Using class distribution: {config.class_distribution}")
    print(f"Using calibration factors: {config.calibration_factors}")
    
    # Get optimizer parameters from config
    label_smoothing = config.get_model_param("label_smoothing", 0.1)
    weight_decay = config.get_model_param("weight_decay", 0.01)
    lr_scheduler_factor = config.get_model_param("lr_scheduler_factor", 0.5)
    lr_scheduler_patience = config.get_model_param("lr_scheduler_patience", 2)
    min_lr = config.get_model_param("min_lr", 1e-6)
    
    # For 3-class classification, use manually defined class weights
    # These weights reflect the original distribution but condensed to 3 classes
    class_weights = torch.tensor([
        config.class_distribution[0],                                        # weight for class 0
        config.class_distribution[1],                                        # weight for class 1
        sum(config.class_distribution[2:])                                   # weight for class 2+
    ], device=device)
    
    # Normalize weights
    normalized_weights = class_weights / class_weights.sum()
    print(f"Using class weights: {normalized_weights}")
    
    # Use label smoothing along with class weights to improve generalization
    criterion = nn.CrossEntropyLoss(
        weight=normalized_weights,
        label_smoothing=label_smoothing  # Add label smoothing to help with overfitting
    )  # Weighted classification loss with smoothing
    
    # Use different learning rates for backbone and classification head
    # Typically, backbone needs smaller learning rate since it's pretrained
    backbone_lr = lr * config.get_model_param("backbone_lr_multiplier", 0.1)
    
    # Create parameter groups with different learning rates
    parameters = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': lr}
    ]
    
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    print(f"Using learning rates - Backbone: {backbone_lr}, Classifier: {lr}")
    
    # Use ReduceLROnPlateau scheduler to reduce LR when validation metrics plateau
    # This helps prevent erratic bouncing around the optimum
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Monitor F1 score which we want to maximize
        factor=lr_scheduler_factor,  # Multiply LR by this factor on plateau
        patience=lr_scheduler_patience,  # Number of epochs with no improvement before reducing LR
        verbose=True,        # Print message when LR is reduced
        min_lr=min_lr        # Don't reduce LR below this value
    )

    # Training metrics
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_balanced_acc": [], "val_f1_macro": []}

    # For early stopping - get patience from config or CLI args
    patience = config.get_model_param("early_stopping_patience")
    # Create early stopping here, outside the epoch loop - only monitor F1 macro
    early_stopping = EarlyStopping(patience=patience, mode="max", verbose=True)
    best_f1_macro = 0
    
    # Create the ModelCheckpoint utility outside the epoch loop so it maintains state across epochs
    checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        metrics=["f1_macro"],  # Only F1 macro is used for saving models now
        mode="max", 
        verbose=True
    )
    
    # Save a copy of the initial model for backup
    best_model = None

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
            
            # HuggingFace models return an object with logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                loss = criterion(logits, targets)
            else:
                loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            
            # Apply gradient clipping to prevent large updates
            gradient_clip_value = config.get_model_param("gradient_clip_value", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Learning rate scheduler will be updated after validation

        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validate using the unified validation function
        metrics = validate(model, val_loader, criterion, device)
        
        # Extract metrics and update history
        val_loss = metrics['loss']
        val_acc = metrics['accuracy']
        val_balanced_acc = metrics['balanced_accuracy']
        val_f1_macro = metrics['f1_macro']
        predictions = metrics['predictions']
        ground_truth = metrics['targets']
        
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_f1_macro"].append(val_f1_macro)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}, Balanced Accuracy: {val_balanced_acc:.2%}, "
            f"F1 Macro: {val_f1_macro:.2%}"
        )
        
        # Update learning rate scheduler based on F1 score
        scheduler.step(val_f1_macro)
        
        # Check if any metric has improved and save the model if needed
        improved = checkpoint.check_improvement(
            metrics_dict=metrics,
            model=model,
            model_type=model_type
        )
        
        # If model improved, keep a copy in memory too
        if improved:
            # Create a deep copy of the model
            best_model = ModelFactory.load_model(
                output_path / f"receipt_counter_{model_type}_best.pth",
                mode="eval",
                model_type=model_type
            )
        
        # Use the already created EarlyStopping utility to decide whether to stop training
        should_stop = early_stopping.check_improvement(val_f1_macro)
        if should_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    ModelFactory.save_model(model, output_path / f"receipt_counter_{model_type}_final.pth")

    # Save training history
    pd.DataFrame(history).to_csv(
        output_path / f"{model_type}_classification_history.csv", index=False
    )

    # Plot training curves using the unified function
    plot_training_curves(
        history,
        output_path=output_path / f"{model_type}_classification_curves.png"
    )
    
    # We should already have the best model in memory
    print("\nEvaluating the best model...")
    
    # If we have the best model in memory, use it
    if best_model is not None:
        print("Using the best model that was saved during training")
        best_model = best_model.to(device)
    else:
        # As a fallback, try to load from disk
        best_model_path = output_path / f"receipt_counter_{model_type}_best.pth"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            best_model = ModelFactory.load_model(best_model_path, model_type=model_type).to(device)
        else:
            # If best model doesn't exist for some reason, use the final model
            print("WARNING: Best model not found. Using final model instead.")
            best_model = model
    
    # Always evaluate the best model
    best_metrics = validate(best_model, val_loader, criterion, device)
    
    # Plot confusion matrix for best model
    accuracy, balanced_accuracy = plot_confusion_matrix(
        best_metrics['predictions'],
        best_metrics['targets'],
        output_path=output_path / f"{model_type}_classification_results.png",
    )
    
    # Get F1 score from best model
    f1_macro = best_metrics['f1_macro']

    print("\nBest Model Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Macro: {f1_macro:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    
    # Print the same format as during training for easy comparison
    print(f"For comparison with training output: F1 Macro: {f1_macro:.2%}")
    
    # ALWAYS return the best model - this is critical!
    model = best_model

    print(f"\nTraining complete! Models saved to {output_path}/")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a SwinV2 model for receipt counting as classification"
    )
    
    # Data input options
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        "--train_csv", "-tc",
        default="receipt_dataset/train.csv",
        help="Path to CSV file containing training data",
    )
    data_group.add_argument(
        "--train_dir", "-td",
        default="receipt_dataset/train",
        help="Directory containing training images",
    )
    data_group.add_argument(
        "--val_csv", "-vc",
        default="receipt_dataset/val.csv",
        help="Path to CSV file containing validation data",
    )
    data_group.add_argument(
        "--val_dir", "-vd",
        default="receipt_dataset/val",
        help="Directory containing validation images",
    )
    data_group.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation during training"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        "--model_type", type=str, choices=["swinv2", "swinv2-large"],
        help="Type of SwinV2 model to use (default from config)"
    )
    training_group.add_argument(
        "--offline", action="store_true",
        help="Use offline mode with locally downloaded model weights"
    )
    training_group.add_argument(
        "--pretrained_model_dir", type=str,
        help="Directory containing pre-downloaded model weights (used with --offline)"
    )
    training_group.add_argument(
        "--epochs", "-e", type=int,
        help="Number of training epochs (default from config)"
    )
    training_group.add_argument(
        "--batch_size", "-b", type=int,
        help="Batch size for training (uses config default if not specified)"
    )
    training_group.add_argument(
        "--patience", "-p", type=int,
        help="Early stopping patience (epochs without improvement before stopping)"
    )
    training_group.add_argument(
        "--lr", "-l", type=float,
        help="Learning rate (default from config)"
    )
    training_group.add_argument(
        "--backbone_lr_multiplier", "-blrm", type=float,
        help="Multiplier for backbone learning rate relative to classifier (default from config)"
    )
    training_group.add_argument(
        "--weight_decay", "-wd", type=float,
        help="Weight decay for optimizer (default: from config)"
    )
    training_group.add_argument(
        "--label_smoothing", "-ls", type=float,
        help="Label smoothing factor (default: from config)"
    )
    training_group.add_argument(
        "--grad_clip", "-gc", type=float,
        help="Gradient clipping max norm (default: from config)"
    )
    training_group.add_argument(
        "--output_dir", "-o",
        default="models",
        help="Directory to save trained model and results",
    )
    training_group.add_argument(
        "--config", "-c",
        help="Path to configuration JSON file with class distribution and calibration factors",
    )
    training_group.add_argument(
        "--resume", "-r",
        help="Resume training from checkpoint file"
    )
    training_group.add_argument(
        "--binary", "-bin", action="store_true",
        help="Train as binary classification (multiple receipts or not)"
    )
    training_group.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration without actual training"
    )
    
    # Class distribution
    training_group.add_argument(
        "--class_dist", 
        help="Comma-separated class distribution (e.g., '0.3,0.2,0.2,0.1,0.1,0.1')"
    )
    
    # Reproducibility options
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument(
        "--seed", "-s", type=int,
        help="Random seed for reproducibility"
    )
    repro_group.add_argument(
        "--deterministic", "-d", action="store_true",
        help="Enable deterministic mode for reproducibility (may reduce performance)"
    )

    args = parser.parse_args()

    # Get configuration singleton
    config = get_config()
    
    # Load configuration if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Override class distribution if provided (and not in binary mode)
    if args.class_dist and not args.binary:
        try:
            dist = [float(x) for x in args.class_dist.split(',')]
            if len(dist) != 5 and not args.binary:
                raise ValueError("Class distribution must have exactly 5 values for multiclass mode")
            config.update_class_distribution(dist)
            print(f"Using custom class distribution: {dist}")
        except Exception as e:
            print(f"Error parsing class distribution: {e}")
            print("Using default class distribution")
    
    # If binary mode is specified, it overrides any class_dist setting
    if args.binary:
        # Binary mode configuration will be handled in train_model
        print("Binary mode specified - will train for 'multiple receipts or not' classification")
    
    # Override hyperparameters if specified
    if args.weight_decay is not None:
        config.update_model_param("weight_decay", args.weight_decay)
        print(f"Using custom weight decay: {args.weight_decay}")
        
    if args.label_smoothing is not None:
        config.update_model_param("label_smoothing", args.label_smoothing)
        print(f"Using custom label smoothing: {args.label_smoothing}")
        
    # Set backbone learning rate multiplier if provided
    if args.backbone_lr_multiplier is not None:
        config.update_model_param("backbone_lr_multiplier", args.backbone_lr_multiplier)
        print(f"Using backbone learning rate multiplier: {args.backbone_lr_multiplier}")
    
    # Set gradient clipping if specified
    if args.grad_clip is not None:
        config.update_model_param("gradient_clip_value", args.grad_clip)
        print(f"Using gradient clipping max norm: {args.grad_clip}")
        
    # Set early stopping patience if provided
    if args.patience is not None:
        config.update_model_param("early_stopping_patience", args.patience)
        print(f"Using early stopping patience: {args.patience}")
        
    # Set reproducibility parameters if provided
    if args.seed is not None:
        config.update_model_param("random_seed", args.seed)
        print(f"Using user-specified random seed: {args.seed}")
        
    if args.deterministic is not None:
        config.update_model_param("deterministic_mode", args.deterministic)
        mode_str = "enabled" if args.deterministic else "disabled"
        print(f"Deterministic mode {mode_str} by user")

    # Validate that files exist
    train_csv_path = Path(args.train_csv)
    train_dir_path = Path(args.train_dir)
    
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV file not found: {train_csv_path}")
    if not train_dir_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir_path}")

    # Optional validation files
    val_csv_path = Path(args.val_csv) if args.val_csv else None
    val_dir_path = Path(args.val_dir) if args.val_dir else None
    
    # Check if validation files exist when provided
    val_csv = args.val_csv if (val_csv_path and val_csv_path.exists()) else None
    val_dir = args.val_dir if (val_dir_path and val_dir_path.exists()) else None
    
    # Check if we're resuming training
    resume_checkpoint = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        resume_checkpoint = args.resume
        print(f"Resuming training from checkpoint: {resume_path}")
    
    # If dry run, just print configuration and exit
    if args.dry_run:
        # Get config values for any None args
        model_type = args.model_type if args.model_type is not None else config.get_model_param("model_type")
        epochs = args.epochs if args.epochs is not None else config.get_model_param("epochs")
        batch_size = args.batch_size if args.batch_size is not None else config.get_model_param("batch_size")
        lr = args.lr if args.lr is not None else config.get_model_param("learning_rate")
        backbone_lr_multiplier = args.backbone_lr_multiplier if args.backbone_lr_multiplier is not None else config.get_model_param("backbone_lr_multiplier", 0.1)
        augment = not args.no_augment if args.no_augment is not None else config.get_model_param("data_augmentation")
        
        print("\n=== DRY RUN - CONFIGURATION VALIDATION ===")
        print(f"Model type: {model_type}")
        print(f"Training data: {args.train_csv} ({args.train_dir})")
        print(f"Validation data: {val_csv} ({val_dir})")
        print(f"Binary mode: {args.binary}")
        print(f"Data augmentation: {'disabled' if not augment else 'enabled'}")
        print(f"Offline mode: {args.offline}")
        if args.offline and args.pretrained_model_dir:
            print(f"Pre-downloaded model dir: {args.pretrained_model_dir}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate - Classifier: {lr}, Backbone: {lr * backbone_lr_multiplier}")
        print(f"Output directory: {args.output_dir}")
        print(f"Reproducibility: seed={config.get_model_param('random_seed')}, deterministic={config.get_model_param('deterministic_mode')}")
        print(f"Class distribution: {config.class_distribution}")
        print(f"Weight decay: {config.get_model_param('weight_decay')}")
        print(f"Label smoothing: {config.get_model_param('label_smoothing')}")
        print(f"Early stopping patience: {config.get_model_param('early_stopping_patience')}")
        if resume_checkpoint:
            print(f"Resuming from: {resume_checkpoint}")
        print("=== CONFIGURATION VALID ===\n")
        return

    train_model(
        args.train_csv,
        args.train_dir,
        val_csv,
        val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        binary=args.binary,
        augment=(not args.no_augment),  # Pass augmentation flag to train_model
        resume_checkpoint=resume_checkpoint if args.resume else None,
        model_type=args.model_type,
        offline=args.offline,
        pretrained_model_dir=args.pretrained_model_dir,
    )


if __name__ == "__main__":
    main()