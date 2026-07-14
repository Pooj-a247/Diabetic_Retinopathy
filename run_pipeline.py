import torch
import numpy as np
from PIL import Image
import os

def run_inference():
    # 1. Load the compiled models
    print("📂 Loading compiled models...")
    dae = torch.jit.load("dae_retina_compiled.pt")
    unet = torch.jit.load("unet_retina_compiled.pt")
    dae.eval()
    unet.eval()
    
    # 2. Check for a test image, otherwise create dummy data
    image_path = "test_retina.png" # Replace with a real retina image if you have one!
    if os.path.exists(image_path):
        print(f"🖼️ Loading test image: {image_path}...")
        img = Image.open(image_path).convert('L') # Convert to Grayscale
        img = img.resize((512, 512))
        input_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    else:
        print("⚠️ No test image found. Generating a simulated grayscale retina tensor (512x512)...")
        input_tensor = torch.rand(1, 1, 512, 512)

    # 3. Run Pipeline
    with torch.no_grad():
        print("🧼 Step 1: Running Denoising Autoencoder (DAE)...")
        denoised_output = dae(input_tensor)
        
        print("👁️ Step 2: Running U-Net Segmentation...")
        segmented_output = unet(denoised_output)
        
        # Apply sigmoid to get probabilities (0 to 1)
        segmentation_mask = torch.sigmoid(segmented_output)
        
    print("💾 Saving pipeline output...")
    # Convert output back to an image and save it
    mask_np = (segmentation_mask.squeeze().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save("pipeline_output.png")
    
    print("🎉 Success! The pipeline executed perfectly. Output saved as 'pipeline_output.png'.")

if __name__ == "__main__":
    run_pipeline = run_inference()
