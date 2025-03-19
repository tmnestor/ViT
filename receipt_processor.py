import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

class ReceiptProcessor:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
    def preprocess_image(self, image_path):
        """Process a scanned document image for receipt counting."""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        preprocessed = self.transform(image=img)["image"]
        
        # Add batch dimension
        return preprocessed.unsqueeze(0)
        
    def enhance_scan_quality(self, image_path, output_path=None):
        """Enhance scanned image quality for better receipt detection."""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        if output_path:
            cv2.imwrite(output_path, opening)
            
        return opening