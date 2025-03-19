import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
from receipt_counter import ReceiptCounter
from receipt_processor import ReceiptProcessor

# Custom dataset for receipt counting
class ReceiptDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or ReceiptProcessor().transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        # Receipt count as target
        count = self.data.iloc[idx, 1]
        return image, torch.tensor(count, dtype=torch.float)

def train_model(data_path, img_dir, epochs=15, batch_size=16, lr=3e-5):
    # Initialize dataset and loader
    dataset = ReceiptDataset(data_path, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReceiptCounter(pretrained=True).to(device)
    
    # Loss and optimizer with increased regularization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)  # Increased weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, counts in dataloader:
            images, counts = images.to(device), counts.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), counts)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
    
    # Save model
    model.save("receipt_counter_swin_tiny.pth")
    print("Training complete!")
    
    return model

if __name__ == "__main__":
    train_model("receipt_data.csv", "receipt_images/")