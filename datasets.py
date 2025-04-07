import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from receipt_processor import ReceiptProcessor
from config import get_config


class ReceiptDataset(Dataset):
    """
    Unified dataset class for receipt counting. 
    Handles both regular receipt images and collage images consistently.
    """
    def __init__(self, csv_file, img_dir, transform=None, augment=False, binary=False):
        """
        Initialize a receipt dataset.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            augment: Whether to apply data augmentation (used for training)
            binary: Whether to use binary classification mode (0 vs 1+ receipts)
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.root_dir = os.path.dirname(self.img_dir)
        self.binary = binary  # Flag for binary classification
        
        # Get configuration parameters
        config = get_config()
        self.image_size = config.get_model_param("image_size", 224)
        
        # Get normalization parameters from config
        self.mean = np.array(config.get_model_param("normalization_mean", [0.485, 0.456, 0.406]))
        self.std = np.array(config.get_model_param("normalization_std", [0.229, 0.224, 0.225]))
        
        # Use provided transform or create from receipt processor
        self.transform = transform or ReceiptProcessor(augment=augment).transform
        
        # Print the first few file names in the dataset for debugging
        print(f"First few files in dataset: {self.data.iloc[:5, 0].tolist()}")
        print(f"Checking for image files in: {self.img_dir} and parent dir")
        if binary:
            print("Using binary classification mode (0 vs 1+ receipts)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        
        # Only use the primary img_dir - no fallbacks to prevent data leakage
        image_path = os.path.join(self.img_dir, filename)
        
        # Check if the image exists in the specified directory
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Log the error but don't fallback to other directories
            print(f"Error: Image {filename} not found at {image_path}")
            image = None
        
        if image is None:
            # Fallback to using a blank image rather than crashing
            print(f"Warning: Could not find image {filename} in any potential location.")
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Convert to numpy array for transformation
        image_np = np.array(image)
        
        # Apply transformations if available
        if self.transform:
            # The transform from receipt_processor.py is a torchvision transform (not albumentations)
            # that works directly with PIL images, not with numpy arrays + keywords
            image_tensor = self.transform(image)
        else:
            # Manual resize and normalization if no transform provided
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = (image_np - self.mean) / self.std
            image_tensor = torch.tensor(image_np).permute(2, 0, 1)

        # Receipt count as target class
        count = int(self.data.iloc[idx, 1])
        
        if self.binary:
            # Convert to binary classification (0 vs 1+ receipts)
            binary_label = 1 if count > 0 else 0
            return image_tensor, torch.tensor(binary_label, dtype=torch.long)
        else:
            # 3-class classification (0, 1, 2+ receipts)
            # Map all counts of 2 or more to class 2
            if count >= 2:
                count = 2
            return image_tensor, torch.tensor(count, dtype=torch.long)


def create_data_loaders(
    train_csv, 
    train_dir, 
    val_csv=None, 
    val_dir=None, 
    batch_size=None, 
    augment_train=True, 
    binary=False,
    train_val_split=0.8
):
    """
    Create data loaders for training and validation.
    
    Args:
        train_csv: Path to CSV file containing training data
        train_dir: Directory containing training images
        val_csv: Path to CSV file containing validation data (optional)
        val_dir: Directory containing validation images (optional)
        batch_size: Batch size for training and validation
        augment_train: Whether to apply data augmentation to training set
        binary: Whether to use binary classification mode
        train_val_split: Proportion of training data to use for training (if no val_csv provided)
        
    Returns:
        tuple: (train_loader, val_loader, num_train_samples, num_val_samples)
    """
    # Get configuration
    config = get_config()
    num_workers = config.get_model_param("num_workers", 4)
    # Use batch_size from config if not explicitly provided
    if batch_size is None:
        batch_size = config.get_model_param("batch_size", 8)
    
    # Ensure dataset directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if val_dir and not os.path.exists(val_dir):
        print(f"Warning: Validation directory not found: {val_dir}. Using training directory split instead.")
        val_dir = None
        val_csv = None  # Reset val_csv to force train/val split
    
    # If separate validation set is provided
    if val_csv and val_dir and os.path.exists(val_csv):
        # Create datasets with appropriate augmentation settings
        train_dataset = ReceiptDataset(train_csv, train_dir, augment=augment_train, binary=binary)
        val_dataset = ReceiptDataset(val_csv, val_dir, augment=False, binary=binary)  # No augmentation for validation
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        return train_loader, val_loader, len(train_dataset), len(val_dataset)
    
    # If no separate validation set, split the training set
    else:
        # Create datasets with and without augmentation
        train_data_with_aug = ReceiptDataset(train_csv, train_dir, augment=augment_train, binary=binary)
        val_data_no_aug = ReceiptDataset(train_csv, train_dir, augment=False, binary=binary)
        
        # Calculate split sizes
        dataset_size = len(train_data_with_aug)
        train_size = int(train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        # Generate random indices for the split and ensure they don't overlap
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create data loaders with appropriate samplers
        train_loader = DataLoader(
            train_data_with_aug, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data_no_aug, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
        )
        
        print(f"Split {dataset_size} samples into {train_size} training and {val_size} validation samples")
        return train_loader, val_loader, train_size, val_size