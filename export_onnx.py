import torch
import os

# Import the dynamic shell classes we created earlier
from compile_model import DynamicModelShell
from compile_dae import DynamicDAEShell

def export_to_onnx():
    dae_weights = "dae_retina.pth"
    unet_weights = "unet_retina.pth"

    # 1. Verify weights exist
    if not os.path.exists(dae_weights) or not os.path.exists(unet_weights):
        print("❌ Error: One or both of the raw .pth weights files are missing!")
        return

    # 2. Setup standard dummy input [1, 1, 512, 512]
    dummy_input = torch.randn(1, 1, 512, 512)

    # 3. Instantiate and Load DAE
    print("📂 Loading DAE raw weights and building shell...")
    dae_state = torch.load(dae_weights, map_location=torch.device('cpu'))
    dae_model = DynamicDAEShell(dae_state)
    dae_model.load_state_dict(dae_state)
    dae_model.eval()

    # 4. Export DAE to ONNX
    print("📤 Exporting Denoising Autoencoder (DAE) to ONNX...")
    torch.onnx.export(
        dae_model, 
        dummy_input, 
        "dae_retina.onnx", 
        export_params=True, 
        opset_version=17,  # Explicitly using opset 17 for high TensorRT compatibility
        do_constant_folding=True,
        input_names=['input_dae'], 
        output_names=['output_dae']
    )
    print("✅ DAE successfully exported as 'dae_retina.onnx'\n")

    # 5. Instantiate and Load U-Net
    print("📂 Loading U-Net raw weights and building shell...")
    unet_state = torch.load(unet_weights, map_location=torch.device('cpu'))
    unet_model = DynamicModelShell(unet_state)
    unet_model.load_state_dict(unet_state)
    unet_model.eval()

    # 6. Export U-Net to ONNX
    print("📤 Exporting U-Net to ONNX...")
    torch.onnx.export(
        unet_model, 
        dummy_input, 
        "unet_retina.onnx", 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True,
        input_names=['input_unet'], 
        output_names=['output_unet']
    )
    print("✅ U-Net successfully exported as 'unet_retina.onnx'\n")
    print("🎉 All models successfully exported to ONNX format!")

if __name__ == "__main__":
    export_to_onnx()
