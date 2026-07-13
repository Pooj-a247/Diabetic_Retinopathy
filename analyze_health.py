import os
import numpy as np
import skimage.io as io
from run_fractal_pipeline import box_count

MASK_DIR = "unet_output"

def analyze():
    if not os.path.exists(MASK_DIR):
        print(f"Error: Target mask folder not found at {MASK_DIR}")
        return
        
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('_mask.jpg')])
    if not mask_files:
        print("No processed U-Net mask files found! Make sure to run predict_unet.py first.")
        return

    data = []
    for fname in mask_files:
        mask = io.imread(os.path.join(MASK_DIR, fname), as_gray=True)
        d_val = box_count(mask)
        data.append((fname, d_val))

    dims = [d for _, d in data]
    mean_d = np.mean(dims)
    std_d = np.std(dims)
    lower_bound = mean_d - (1.5 * std_d)

    print("=" * 65)
    print("           RETINAL STRUCTURAL ANOMALY DETECTOR REPORT          ")
    print("=" * 65)
    print(f"Dataset Baseline Mean D : {mean_d:.4f} (StdDev: {std_d:.4f})")
    print(f"Normal Complexity Threshold: >= {lower_bound:.4f}")
    print("-" * 65)
    print(f"{'Mask Name':<25} | {'Fractal Dim':<15} | {'Status':<15}")
    print("-" * 65)

    for fname, d_val in data:
        status = "NORMAL" if d_val >= lower_bound else "ANOMALY (Sparse)"
        print(f"{fname:<25} | {d_val:<15.4f} | {status:<15}")
    print("=" * 65)

if __name__ == "__main__":
    analyze()
