import torch

def convert_yolo_to_torchscript(model_path, output_path):
    """Simple YOLO to TorchScript converter"""
    
    # Load the model with weights_only=False to handle YOLOv8 models
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    # Handle different YOLO formats
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
            print("Extracted model from checkpoint dict")
        elif 'state_dict' in model:
            print("ERROR: This appears to be a state dict. You need the model architecture to convert it.")
            return False
        else:
            print("Unknown model format")
            return False
    
    # Ensure model is in eval mode and convert to float32
    model.eval()
    model.float()  # Convert model to float32 to avoid half/float tensor issues
    
    # Create dummy input (YOLO typically uses 640x640)
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    
    print("Converting to TorchScript...")
    
    # Try to simplify the model for TorchScript compatibility
    try:
        # For YOLOv8, we need to handle the export mode
        if hasattr(model, 'export'):
            model.export = True
        if hasattr(model, 'mode'):
            model.mode = 'export'
        
        # Disable any training-specific features
        for m in model.modules():
            if hasattr(m, 'training'):
                m.training = False
            if hasattr(m, 'export'):
                m.export = True
        
        print("Attempting torch.jit.trace...")
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        traced_model.save(output_path)
        print(f"✅ Converted successfully: {output_path}")
        
        # Verify the conversion worked
        print("Verifying conversion...")
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            test_output = loaded_model(dummy_input)
            print(f"✅ Verification passed. Output shape: {test_output.shape if hasattr(test_output, 'shape') else type(test_output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trace conversion failed: {e}")
        
        # Alternative approach: try to export using ultralytics built-in export
        try:
            print("Trying ultralytics export method...")
            from ultralytics import YOLO
            
            # Load using ultralytics and export
            yolo_model = YOLO(model_path)
            yolo_model.export(format='torchscript', optimize=True)
            
            # The exported file will have a different name, let's find it
            import os
            export_path = model_path.replace('.pt', '.torchscript')
            if os.path.exists(export_path):
                # Move to desired output path
                import shutil
                shutil.move(export_path, output_path)
                print(f"✅ Exported using ultralytics: {output_path}")
                return True
            else:
                print("❌ Ultralytics export didn't create expected file")
                return False
                
        except Exception as e2:
            print(f"❌ Ultralytics export failed: {e2}")
            
            # Last resort: try manual simplification
            try:
                print("Trying manual model simplification...")
                
                # Create a wrapper that avoids problematic features
                class YOLOWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        return self.model(x)
                
                wrapper = YOLOWrapper(model)
                wrapper.eval()
                
                traced_wrapper = torch.jit.trace(wrapper, dummy_input, strict=False)
                traced_wrapper.save(output_path)
                print(f"✅ Wrapper conversion successful: {output_path}")
                return True
                
            except Exception as e3:
                print(f"❌ All conversion methods failed: {e3}")
                return False

# Usage example:
# convert_yolo_to_torchscript("yolov8n.pt", "yolov8n_torchscript.pt")
convert_yolo_to_torchscript("/home/user/data/last.pt", "/home/user/data/last_ts.pt")