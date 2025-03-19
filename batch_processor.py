import os
import pandas as pd
import argparse
import torch
import glob
from receipt_counter import ReceiptCounter
from receipt_processor import ReceiptProcessor
from tqdm import tqdm

def process_document_archive(model_path, input_dir, output_csv):
    """Process a directory of scanned documents and count receipts in each."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReceiptCounter.load(model_path).to(device)
    processor = ReceiptProcessor()
    
    # Find all image files
    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    
    results = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing documents"):
        try:
            # Preprocess the image
            img_tensor = processor.preprocess_image(img_path).to(device)
            
            # Make prediction
            with torch.no_grad():
                predicted_count = model.predict(img_tensor)
            
            # Round to nearest integer
            count = round(predicted_count)
            
            results.append({
                'document': os.path.basename(img_path),
                'receipt_count': count,
                'raw_prediction': float(predicted_count)
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save results
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Summary
    total_receipts = sum(r['receipt_count'] for r in results)
    print(f"Processed {len(results)} documents")
    print(f"Detected a total of {total_receipts} receipts")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Receipt Counter")
    parser.add_argument("--input_dir", required=True, help="Directory with scanned documents")
    parser.add_argument("--output_csv", required=True, help="Path to save results CSV")
    parser.add_argument("--model", default="receipt_counter_swin_tiny.pth",
                       help="Path to model file")
    
    args = parser.parse_args()
    process_document_archive(args.model, args.input_dir, args.output_csv)