import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from dae_model import DenoisingAutoencoder

# Configuration
IMAGE_DIR = "images"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)

class DAEImageDataset(Dataset):
    def __init__(self, image_dir, size=IMG_SIZE):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        
        # Load and clean dimension shapes
        image_raw = io.imread(img_path)
        image_raw = np.squeeze(image_raw)
        
        if len(image_raw.shape) == 3:
            image = rgb2gray(image_raw).astype(np.float32)
        else:
            image = image_raw.astype(np.float32)
            
        if image.max() > 1.0: 
            image /= 255.0
            
        # Convert to tensor and resize
        clean_tensor = torch.tensor(image).unsqueeze(0)
        clean_tensor = TF.resize(clean_tensor, self.size)
        
        # ⚠️ Generate Synthetic Noise on the fly
        noise = torch.randn_like(clean_tensor) * 0.1 # Adjust 0.1 to change noise intensity
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
        
        return noisy_tensor, clean_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting DAE training on {device}...")
    
    dataset = DAEImageDataset(IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss() # Mean Squared Error calculates pixel-level difference
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for noisy_imgs, clean_imgs in loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "dae_retina.pth")
    print("DAE training finished! Saved weights to 'dae_retina.pth'")

if __name__ == "__main__":
    train()
