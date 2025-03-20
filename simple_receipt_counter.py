# This module is deprecated. Use transformer_swin.py or transformer_vit.py instead.
# Kept for backward compatibility with some utilities.

import torch
import torch.nn as nn
        
# Enhanced function to create a Hugging Face receipt counter
def create_simple_hf_receipt_counter(num_classes=6, model_type="swin"):
    try:
        if model_type == "swin":
            # Use the transformer_swin module
            from transformer_swin import create_swin_transformer
            
            # Create a model with a simple classification head
            model = create_swin_transformer(
                pretrained=True,
                num_classes=num_classes,
                verbose=False
            )
            
            # Optionally simplify the classifier if needed
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            
            return model
        elif model_type == "vit":
            # Use the transformer_vit module
            from transformer_vit import create_vit_transformer
            
            # Create a model with a simple classification head
            model = create_vit_transformer(
                pretrained=True,
                num_classes=num_classes,
                verbose=False
            )
            
            # Optionally simplify the classifier if needed
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            
            return model
        else:
            print(f"Unknown model type: {model_type}")
            return None
            
    except ImportError as e:
        print(f"Transformers library or required module not found: {e}")
        print("Please install with: pip install transformers")
        return None
    except Exception as e:
        print(f"Error creating model: {e}")
        return None