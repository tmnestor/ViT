import os
import pandas as pd
import re
import shutil
from sklearn.model_selection import train_test_split
import argparse

def extract_receipt_count(filename):
    """Extract the receipt count from the collage filename."""
    match = re.search(r'_(\d+)_receipts', filename)
    if match:
        return int(match.group(1))
    return 0

def prepare_dataset(collage_dir, output_dir, test_size=0.2, random_state=42):
    """
    Prepare a dataset from collage images for receipt counting.
    
    Args:
        collage_dir: Directory containing collage images
        output_dir: Directory to save the organized dataset
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        train_csv_path, val_csv_path: Paths to the created CSV files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Collect image paths and receipt counts
    data = []
    for filename in os.listdir(collage_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        receipt_count = extract_receipt_count(filename)
        data.append((filename, receipt_count))
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['filename', 'receipt_count'])
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df['receipt_count'] if len(df['receipt_count'].unique()) < 10 else None
    )
    
    # Copy files and create CSVs
    for df, target_dir, name in [(train_df, train_dir, 'train'), (val_df, val_dir, 'val')]:
        # Copy images
        for _, row in df.iterrows():
            src = os.path.join(collage_dir, row['filename'])
            dst = os.path.join(target_dir, row['filename'])
            shutil.copy2(src, dst)
        
        # Save CSV
        csv_path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Created {name} dataset with {len(df)} images")
    
    return os.path.join(output_dir, 'train.csv'), os.path.join(output_dir, 'val.csv')

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset from collage images")
    parser.add_argument("--collage_dir", default="receipt_collages", 
                        help="Directory containing collage images")
    parser.add_argument("--output_dir", default="receipt_dataset", 
                        help="Directory to save the organized dataset")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    
    args = parser.parse_args()
    
    train_csv, val_csv = prepare_dataset(
        args.collage_dir, args.output_dir, test_size=args.test_size
    )
    
    print(f"\nDataset preparation complete.")
    print(f"Train CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")
    print("\nTo train the model, run:")
    print(f"python train_swin_counter.py --train_csv {train_csv} --train_dir {os.path.join(args.output_dir, 'train')} --val_csv {val_csv} --val_dir {os.path.join(args.output_dir, 'val')}")

if __name__ == "__main__":
    main()