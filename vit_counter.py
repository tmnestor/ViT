import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTReceiptCounter(nn.Module):
    def __init__(self, pretrained=True, num_classes=6):
        super(ViTReceiptCounter, self).__init__()
        
        # Load ViT-Base backbone
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        
        # Replace with classification head (0-5 receipts)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Classification output (0-5 receipts)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    @torch.no_grad()
    def predict_class(self, img_tensor):
        """Return the predicted class (0-5 receipts)"""
        self.eval()
        outputs = self.forward(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
    @torch.no_grad()
    def predict(self, img_tensor):
        self.eval()
        return self.forward(img_tensor).item()
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path):
        model = cls(pretrained=False)
        model.load_state_dict(torch.load(path))
        return model

# Alternative implementation using Hugging Face transformers
def create_hf_vit_receipt_counter(pretrained=True):
    from transformers import ViTForImageClassification
    
    # Load model from Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", 
        num_labels=1,  # Regression task
        ignore_mismatched_sizes=True
    )
    
    # Replace classification head with stronger regularization
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    
    return model