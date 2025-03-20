import os
import pandas as pd
import argparse
import torch
import glob
from transformer_swin import create_swin_transformer, load_swin_model
from receipt_processor import ReceiptProcessor
from tqdm import tqdm

def process_document_archive(model_path, input_dir, output_csv):
    """Process a directory of scanned documents and count receipts in each."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_swin_model(model_path).to(device)
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
                outputs = model(img_tensor)
                # Get logits from model outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get class prediction (0-5)
                probs = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            # The predicted class is directly the count (0-5)
            count = predicted_class
            
            results.append({
                'document': os.path.basename(img_path),
                'receipt_count': count,
                'confidence': float(confidence)
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
    # Show average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.2%}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Receipt Counter")
    parser.add_argument("--input_dir", required=True, help="Directory with scanned documents")
    parser.add_argument("--output_csv", required=True, help="Path to save results CSV")
    parser.add_argument("--model", default="receipt_counter_swin_tiny.pth",
                       help="Path to model file")
    
    args = parser.parse_args()
    process_document_archive(args.model, args.input_dir, args.output_csv)