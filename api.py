from flask import Flask, request, jsonify
import torch
import os
import tempfile
from receipt_counter import ReceiptCounter
from receipt_processor import ReceiptProcessor

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReceiptCounter.load("receipt_counter_swin_tiny.pth").to(device)
processor = ReceiptProcessor()

@app.route('/count_receipts', methods=['POST'])
def count_receipts():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Preprocess image
        img_tensor = processor.preprocess_image(tmp_path).to(device)
        
        # Make prediction
        with torch.no_grad():
            predicted_count = model.predict(img_tensor)
        
        # Round to nearest integer (receipt count must be a whole number)
        count = round(predicted_count)
        confidence = 1.0 - min(abs(predicted_count - count) / 1.0, 0.5) * 2
        
        # Clean up
        os.remove(tmp_path)
        
        return jsonify({
            "receipt_count": count,
            "confidence": float(confidence),
            "raw_prediction": float(predicted_count)
        })
    
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500

# ONNX Export (for faster inference deployment)
def export_to_onnx(model_path="receipt_counter_swin_tiny.pth", 
                  onnx_path="receipt_counter.onnx"):
    model = ReceiptCounter.load(model_path).cpu()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export model
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    # Uncomment to export model
    # export_to_onnx()
    
    # Run API server
    app.run(host='0.0.0.0', port=5000)