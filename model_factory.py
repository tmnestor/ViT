"""
Unified model factory for creating and managing transformer models.
This module centralizes model creation, saving, and loading for both ViT and Swin models.
"""

import torch
import torch.nn as nn
from config import get_config

# SafeBatchNorm1d class to handle single-sample inference
class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that doesn't fail with batch size 1."""
    def forward(self, x):
        if x.size(0) == 1:
            # For single sample, skip normalization
            return x
        return super().forward(x)

class ModelFactory:
    """Factory class for creating and managing transformer models."""
    
    # Model type to HuggingFace model path mapping
    MODEL_PATHS = {
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swin-tiny-patch4-window7-224",
        "swinv2": "microsoft/swinv2-tiny-patch4-window8-256"
    }
    
    @classmethod
    def create_transformer(cls, model_type="swinv2", pretrained=True, num_classes=None, verbose=True, mode="train"):
        """Create a transformer model for receipt counting.
        
        Args:
            model_type: Type of model to create (only "swinv2" is fully supported)
            pretrained: Whether to load pretrained weights from Hugging Face
            num_classes: Number of output classes. If None, will be determined from config
            verbose: Whether to show warnings about weight initialization
            mode: "train" for training mode, "eval" for evaluation mode
            
        Returns:
            Configured transformer model
        """
        import transformers
        
        # Validate model type
        model_type = model_type.lower()
        if model_type not in cls.MODEL_PATHS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_PATHS.keys())}")
        
        # Get configuration
        config = get_config()
        
        # Get number of classes from config if not provided
        if num_classes is None:
            num_classes = len(config.class_distribution)
        
        # Temporarily disable HuggingFace warnings if not verbose
        if not verbose:
            # Save previous verbosity level
            prev_verbosity = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
        
        # Create appropriate model type
        try:
            if model_type == "vit":
                from transformers import ViTForImageClassification
                model = ViTForImageClassification.from_pretrained(
                    cls.MODEL_PATHS["vit"], 
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            elif model_type == "swin":
                from transformers import SwinForImageClassification
                model = SwinForImageClassification.from_pretrained(
                    cls.MODEL_PATHS["swin"], 
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            elif model_type == "swinv2":
                from transformers import Swinv2ForImageClassification
                model = Swinv2ForImageClassification.from_pretrained(
                    cls.MODEL_PATHS["swinv2"], 
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
        finally:
            # Restore previous verbosity if changed
            if not verbose:
                transformers.logging.set_verbosity(prev_verbosity)
        
        # Get classifier architecture parameters from config
        classifier_dims = config.get_model_param("classifier_dims", [768, 512, 256])
        dropout_rates = config.get_model_param("dropout_rates", [0.4, 0.4, 0.3])
        
        # Create unified classifier architecture
        cls._build_classifier(model, classifier_dims, dropout_rates, num_classes)
        
        # Set model mode
        if mode.lower() == "eval":
            model.eval()
        else:
            model.train()
            
        return model
    
    @staticmethod
    def _build_classifier(model, classifier_dims, dropout_rates, num_classes):
        """Build a custom classifier head for the model.
        
        Args:
            model: The transformer model
            classifier_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each layer
            num_classes: Number of output classes
        """
        # Create classification layers
        layers = []
        in_features = model.classifier.in_features
        
        # Build sequential model with the configured parameters
        for i, dim in enumerate(classifier_dims):
            layers.append(nn.Linear(in_features, dim))
            # Use SafeBatchNorm1d to handle single-sample inference
            layers.append(SafeBatchNorm1d(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rates[i]))
            in_features = dim
        
        # Add final classification layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Replace the classifier with our custom architecture
        model.classifier = nn.Sequential(*layers)
    
    @staticmethod
    def save_model(model, path):
        """Save a model's state dictionary to disk.
        
        Args:
            model: The model to save
            path: File path to save to
        """
        torch.save(model.state_dict(), path)
    
    @classmethod
    def load_model(cls, path, num_classes=None, strict=True, mode="eval"):
        """Load a saved SwinV2 transformer model.
        
        Args:
            path: Path to the saved model weights
            num_classes: Number of output classes. If None, will be determined from config
            strict: Whether to enforce strict parameter matching
            mode: "train" for training mode, "eval" for evaluation mode
            
        Returns:
            Loaded SwinV2 model
        """
        # We only support SwinV2 models now
        model_type = "swinv2"
        print(f"Loading model from {path} as a SwinV2 model")
        
        # Load state dict first to inspect
        state_dict = torch.load(path)
        
        # Only SwinV2 models are supported
        model_type = "swinv2"
            
        # Create empty model structure without pretrained weights
        model = cls.create_transformer(
            model_type=model_type,
            pretrained=False,
            num_classes=num_classes,
            verbose=False,
            mode=mode
        )
        
        # We only support SwinV2 models now
        # No need to check for other model types
        
        # Try loading with strict=True first
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded model with strict=True")
        except Exception as e:
            if strict:
                # If strict is required, re-raise the exception
                print(f"Error loading model with strict=True: {e}")
                print("If you're loading a Swin model as SwinV2, incompatibilities are expected.")
                raise e
            else:
                # If strict is not required, try loading with strict=False
                print(f"Warning: Could not load with strict=True: {e}")
                print("Trying again with strict=False...")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("Loaded with strict=False. Model may be missing some weights or have extra parameters.")
                except Exception as e2:
                    print(f"Error even with strict=False: {e2}")
                    print("Model architecture is likely incompatible with the saved weights.")
                    raise e2
                
        return model


# End of ModelFactory class