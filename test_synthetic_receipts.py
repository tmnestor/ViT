import argparse
from pathlib import Path
from create_synthetic_receipts import generate_synthetic_receipt, create_receipt_collage

def main():
    """
    Generate sample synthetic receipts to visually check the results.
    """
    parser = argparse.ArgumentParser(description="Test synthetic receipt generation")
    parser.add_argument("--output_dir", default="receipt_samples", 
                      help="Directory to save sample receipts")
    parser.add_argument("--num_receipts", type=int, default=5,
                      help="Number of individual receipts to create")
    parser.add_argument("--num_collages", type=int, default=3,
                      help="Number of collages to create")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual receipts
    print(f"Generating {args.num_receipts} individual receipts...")
    for i in range(args.num_receipts):
        receipt = generate_synthetic_receipt()
        receipt_path = output_dir / f"receipt_sample_{i+1}.jpg"
        receipt.save(receipt_path, "JPEG", quality=95)
        print(f"Generated {receipt_path}")
    
    # Generate sample collages with different receipt counts
    print(f"\nGenerating {args.num_collages} sample collages...")
    for i in range(args.num_collages):
        receipt_count = min(5, i + 1)  # 1, 2, 3, 4, 5 receipts
        collage, actual_count = create_receipt_collage(receipt_count=receipt_count)
        collage_path = output_dir / f"collage_sample_{i+1}_{actual_count}_receipts.jpg"
        collage.save(collage_path, "JPEG", quality=95)
        print(f"Generated {collage_path} with {actual_count} receipts")
    
    print(f"\nSamples saved to {output_dir}")
    print("Run this script to quickly check the appearance of your synthetic receipts.")
    print("Use create_synthetic_receipts.py for generating your full dataset.")

if __name__ == "__main__":
    main()