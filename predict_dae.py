import os
import torch
import torchvision.transforms.functional as TF
import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from dae_model import DenoisingAutoencoder

# Configuration
IMAGE_DIR = "images" 
OUTPUT_CLEAN_DIR = "clean_images"
IMG_SIZE = (256, 256)

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_CLEAN_DIR, exist_ok=True)
    
    # Load model and weights
    model = DenoisingAutoencoder().to(device)
    if os.path.exists("dae_retina.pth"):
        model.load_state_dict(torch.load("dae_retina.pth", map_location=device))
        print("Loaded DAE weights successfully. Starting image enhancement...")
    else:
        print("Error: 'dae_retina.pth' not found!")
        return

    model.eval()
    
    images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    with torch.no_grad():
        for fname in images:
            img_path = os.path.join(TEST_IMAGE_DIR, fname)
            print(f"Enhancing: {fname}")
            
            # Load and process image
            image_raw = io.imread(img_path)
            image_raw = np.squeeze(image_raw)
            
            if len(image_raw.shape) == 3:
                image = rgb2gray(image_raw).astype(np.float32)
            else:
                image = image_raw.astype(np.float32)
                
            if image.max() > 1.0:
                image /= 255.0
                
            # Prepare tensor for network
            input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device) # Shape: [1, 1, H, W]
            input_tensor = TF.resize(input_tensor, IMG_SIZE)
            
            # Run inference
            clean_tensor = model(input_tensor)
            
            # Convert back to image format
            clean_img = clean_tensor.squeeze().cpu().numpy()
            clean_img = (clean_img * 255).astype(np.uint8)
            
            # Save the clean image
            base_name, _ = os.path.splitext(fname)
            output_path = os.path.join(OUTPUT_CLEAN_DIR, f"{base_name}_clean.jpg")
            io.imsave(output_path, clean_img)
            
    print(f"\nAll enhanced images saved successfully to '{OUTPUT_CLEAN_DIR}'")

if __name__ == "__main__":
    predict()
