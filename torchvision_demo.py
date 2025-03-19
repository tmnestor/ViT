import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load pre-trained ViT model
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Prepare image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and load some example images
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Classes for CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Get a batch of images
images, labels = next(iter(dataloader))

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Display images and predictions
def show_predictions(images, labels):
    # Move images to device and get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
    
    # Show images
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        imshow(images[i])
        correct = "✓" if predicted[i] == labels[i] else "✗"
        plt.title(f"Pred: {classes[predicted[i]]}, True: {classes[labels[i]]} {correct}")
    
    plt.tight_layout()
    plt.savefig('vit_predictions.png')
    plt.show()

# Run demo
if __name__ == "__main__":
    print("Running Vision Transformer demo with CIFAR-10 images")
    show_predictions(images, labels)
    print("Demo complete! Predictions saved to vit_predictions.png")