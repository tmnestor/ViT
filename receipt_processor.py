import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

from config import get_config


class ReceiptProcessor:
    def __init__(self, img_size=None, augment=False):
        # Get configuration
        config = get_config()

        # Use config value or override with provided value
        self.img_size = img_size or config.get_model_param("image_size", 224)

        # Get normalization parameters from config
        mean = config.get_model_param("normalization_mean", [0.485, 0.456, 0.406])
        std = config.get_model_param("normalization_std", [0.229, 0.224, 0.225])

        if augment:
            # Enhanced transform with augmentations for training
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                # Color transforms - brightness/contrast adjustments
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.8),
                # Blur or noise - simulate document scanning artifacts
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.5),
                # Geometric transforms - simulate different scanning angles
                transforms.RandomAffine(
                    degrees=15,                # Rotation range 
                    translate=(0.1, 0.1),      # Translation range
                    scale=(0.8, 1.2),          # Scale range
                ),
                # Convert to tensor first (before RandomErasing which expects tensor)
                transforms.ToTensor(),
                # Simulates occlusions or missing parts
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
                # Normalize
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # Standard transform for evaluation
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def preprocess_image(self, image_path):
        """Process a scanned document image for receipt counting."""
        # Read image using PIL (already in RGB format)
        img = Image.open(image_path)

        # Apply preprocessing (torchvision transforms work directly with PIL images)
        preprocessed = self.transform(img)

        # Add batch dimension
        return preprocessed.unsqueeze(0)

    def enhance_scan_quality(self, image_path, output_path=None):
        """Enhance scanned image quality for better receipt detection."""
        # Open image and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Enhance contrast
        img = ImageOps.autocontrast(img, cutoff=2)
        
        # Apply filter for edge enhancement
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        
        # Convert to binary using threshold
        # Create a point function to threshold the image
        threshold = 200  # Adjust this value to get proper thresholding
        img = img.point(lambda p: 255 if p > threshold else 0)
        
        # Apply morphological operations to reduce noise
        # For opening, we first erode then dilate to remove small noise
        img = img.filter(ImageFilter.MinFilter(3))  # Similar to erosion
        img = img.filter(ImageFilter.MaxFilter(3))  # Similar to dilation
        
        # Convert to numpy array for return consistency
        result = np.array(img)
        
        if output_path:
            img.save(output_path)
        
        return result
