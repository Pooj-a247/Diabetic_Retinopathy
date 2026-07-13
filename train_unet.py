import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from unet_model import UNet
import skimage.io as io
from skimage.color import rgb2gray  # <-- ADD THIS LINE
import numpy as np

class RetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        
        # Load image and immediately strip out wrapper dimensions (like shape 1)
        image_raw = io.imread(img_path)
        image_raw = np.squeeze(image_raw) 
        
        # Now safely check if it is truly a 3D color image (H, W, 3)
        if len(image_raw.shape) == 3:
            image = rgb2gray(image_raw).astype(np.float32)
        else:
            image = image_raw.astype(np.float32)
            
        # Do the exact same safe processing for the mask
        mask_raw = io.imread(mask_path)
        mask_raw = np.squeeze(mask_raw)
        
        if len(mask_raw.shape) == 3:
            mask = rgb2gray(mask_raw).astype(np.float32)
        else:
            mask = mask_raw.astype(np.float32)
            
        # Normalize to 0-1 range if needed
        if image.max() > 1.0: image /= 255.0
        if mask.max() > 1.0: mask /= 255.0
        
        mask = (mask > 0.5).astype(np.float32) # Convert to strict binary
        
        # Convert to PyTorch Tensors
        image = torch.tensor(image).unsqueeze(0) # Shape: [1, H, W]
        mask = torch.tensor(mask).unsqueeze(0)   # Shape: [1, H, W]
        
        # Resize to a consistent square dimension
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        
        # ⚠️ CRUCIAL FIX: Threshold AFTER resizing to eliminate interpolation artifacts
        mask = (mask > 0.5).to(torch.float32)
        
        return image, mask

# Configuration
IMAGE_DIR = "clean_images"  # Pointing to DAE outputs 
MASK_DIR = "ground_truth"
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = RetinalDataset(IMAGE_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for pixel segmentation
    
    print(f"Starting training on {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            # Backward pass & weight optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss/len(loader):.4f}")
        
    # Save the trained weights
    torch.save(model.state_dict(), "unet_retina.pth")
    print("Training finished! Saved model weights to 'unet_retina.pth'")

if __name__ == "__main__":
    train()
