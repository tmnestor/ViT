import os
from pathlib import Path
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from datetime import datetime, timedelta
from create_synthetic_receipts import generate_synthetic_receipt

def create_varied_receipt_samples(num_samples=50, output_dir="synthetic_receipts/samples"):
    """
    Create a variety of receipt samples with different sizes, layouts, and content.
    
    Args:
        num_samples: Number of receipt samples to generate
        output_dir: Directory to save the receipt samples
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get existing samples
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("receipt_sample_")]
    existing_nums = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
    start_num = max(existing_nums) + 1 if existing_nums else 1
    
    print(f"Generating {num_samples} receipt samples starting from {start_num}...")
    
    # Store names for variety
    store_names = [
        "QUICKMART", "GROCERY WORLD", "VALUESTORE", "MEGA SHOP", 
        "DAILY NEEDS", "FRESH MARKET", "CITY MART", "MINI SHOP",
        "SUPER SAVE", "FOOD PLUS", "CORNER STORE", "EXPRESS MART",
        "SUPERCENTER", "FAMILY MARKET", "NEIGHBORHOOD SHOP", "DISCOUNT MART",
        "VILLAGE STORE", "STAR MARKET", "PRIME GROCER", "QUALITY FOODS",
        "COUNTRY MARKET", "URBAN PANTRY", "BARGAIN BAZAAR", "VALUE MART"
    ]
    
    # Payment methods for variety
    payment_methods = [
        "CASH", "VISA", "MASTERCARD", "DEBIT", "AMEX", "DISCOVER", "GIFT CARD", 
        "APPLE PAY", "GOOGLE PAY", "VENMO", "PAYPAL", "CHECK", "STORE CREDIT", "EBT"
    ]
    
    # Create varied receipt samples
    for i in range(num_samples):
        sample_num = start_num + i
        
        # Vary receipt dimensions
        if i % 5 == 0:
            # Narrow receipt
            width = random.randint(300, 350)
            height = random.randint(800, 1000)
        elif i % 5 == 1:
            # Wide receipt
            width = random.randint(450, 500)
            height = random.randint(700, 900)
        elif i % 5 == 2:
            # Short receipt
            width = random.randint(350, 450)
            height = random.randint(600, 700)
        elif i % 5 == 3:
            # Long receipt
            width = random.randint(350, 450)
            height = random.randint(1000, 1200)
        else:
            # Medium receipt
            width = random.randint(350, 450)
            height = random.randint(700, 900)
        
        # Generate receipt with varied parameters
        receipt = generate_synthetic_receipt(width=width, height=height)
        
        # Save the receipt
        output_path = os.path.join(output_dir, f"receipt_sample_{sample_num}.jpg")
        receipt.save(output_path, "JPEG", quality=95)
        print(f"Generated {output_path}")
    
    print(f"Created {num_samples} varied receipt samples in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate varied receipt samples")
    parser.add_argument("--num_samples", type=int, default=40,
                      help="Number of receipt samples to generate")
    parser.add_argument("--output_dir", default="synthetic_receipts/samples",
                      help="Directory to save receipt samples")
    
    args = parser.parse_args()
    create_varied_receipt_samples(args.num_samples, args.output_dir)