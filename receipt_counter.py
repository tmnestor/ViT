import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class ReceiptCounter(nn.Module):
    def __init__(self, pretrained=True, num_classes=6):
        super(ReceiptCounter, self).__init__()
        
        # Load Swin-Tiny backbone
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        
        # Replace with classification head (0-5 receipts)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
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
        """For backward compatibility - use predict_class instead"""
        self.eval()
        outputs = self.forward(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path):
        model = cls(pretrained=False)
        model.load_state_dict(torch.load(path))
        return model

# Alternative implementation using Hugging Face transformers
def create_hf_receipt_counter(pretrained=True):
    from transformers import SwinForImageClassification
    
    # Load model from Hugging Face
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", 
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