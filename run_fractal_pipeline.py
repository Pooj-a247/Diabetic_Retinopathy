import os
import numpy as np
import skimage.io as io

# Configuration
MASK_DIR = "unet_output"

def box_count(binary_mask):
    """
    Standard Minkowski-Bouligand Box-Counting algorithm implementation.
    """
    # Ensure binary format
    p = binary_mask > 0
    
    # Check bounding dimensions
    if not np.any(p):
        return 0.0
        
    # Determine maximum size dimension for grids
    max_dim = max(p.shape)
    n = 2**int(np.ceil(np.log2(max_dim)))
    
    # Pad image to fit perfectly inside the 2^n square box layout
    padded = np.zeros((n, n), dtype=bool)
    padded[:p.shape[0], :p.shape[1]] = p
    
    # Calculate box sizes scaled down exponentially by powers of 2
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    
    for box_size in sizes:
        # Reshape into blocks of size (box_size, box_size) and sum target intersections
        blocks = (padded.reshape(n // box_size, box_size, n // box_size, box_size)
                        .swapaxes(1, 2)
                        .reshape(-1, box_size, box_size))
        # Keep track of active boxes intersecting any pixels
        intersecting_boxes = np.sum(np.any(blocks, axis=(1, 2)))
        counts.append(intersecting_boxes)
        
    # Fit coefficients via linear regression (y = m * x + c)
    coefficients = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return coefficients[0] # Returns slope (Fractal Dimension D)

def process_pipeline():
    if not os.path.exists(MASK_DIR):
        print(f"Error: Target mask folder not found at {MASK_DIR}")
        return

    print("=" * 60)
    print("      RETINAL FRACTAL DIMENSION EXTRACTION PIPELINE        ")
    print("=" * 60)
    print(f"{'Generated Mask File':<35} | {'Fractal Dimension (D)':<20}")
    print("-" * 60)

    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('_mask.jpg')])
    
    if not mask_files:
        print("No processed U-Net mask files found! Make sure to run predict_unet.py first.")
        return

    for fname in mask_files:
        mask_path = os.path.join(MASK_DIR, fname)
        
        # Safely ingest output file as a grayscale grid
        mask_raw = io.imread(mask_path, as_gray=True)
        
        # Extract fractal dimension index via box counting
        d_value = box_count(mask_raw)
        
        print(f"{fname:<35} | D = {d_value:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    process_pipeline()
