"""
Simple receipt counter module using the ModelFactory.
This is a simplified model creator for basic use cases.
"""

import torch
import torch.nn as nn
from model_factory import ModelFactory, SafeBatchNorm1d
        
def create_simple_hf_receipt_counter(num_classes=6, model_type="swin"):
    """
    Create a simplified transformer model for receipt counting.
    
    Args:
        num_classes: Number of output classes
        model_type: Either "vit" or "swin"
    
    Returns:
        A configured transformer model with simplified classifier head
    """
    try:
        # Create a model using the factory
        model = ModelFactory.create_transformer(
            model_type=model_type,
            pretrained=True,
            num_classes=num_classes,
            verbose=False
        )
        
        # Simplify the classifier for basic use
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            SafeBatchNorm1d(512),  # Use SafeBatchNorm1d for single-sample inference
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        return model
            
    except ImportError as e:
        print(f"Required libraries not found: {e}")
        print("Please install with: pip install torch transformers")
        return None
    except Exception as e:
        print(f"Error creating model: {e}")
        return None