import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import FastSaliencyUNet
from predict import load_model, run_pipeline

def make_overlay(img_bgr, mask_01, color=(0, 255, 0), alpha=0.45):
    """
    Creates a semi-transparent colored overlay of a binary mask onto a BGR image.
    Color is in BGR format (e.g., (0, 255, 0) for vibrant green).
    """
    overlay = img_bgr.copy()
    colored_mask = np.zeros_like(img_bgr)
    for c in range(3):
        colored_mask[:, :, c] = mask_01 * color[c]
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1.0, 0, overlay)
    # Highlight boundary line for crisp edge perception
    from metrics import extract_boundary
    bnd = extract_boundary(mask_01)
    overlay[bnd] = [255, 255, 255] # White border line
    return overlay
def generate_captured_overlays(weights_path="parameters.pth", out_dir="report/figures"):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(weights_path)
    
    captured_items = ["bag", "bottle", "cats", "cycle", "flower", "shoes", "watch"]
    descriptions = {
        "bag": "Hostel Backpack (Indoor, Bed Pattern Background, Near Object)",
        "bottle": "Thermos Flask (Indoor Desk, Cluttered Background & Bottles)",
        "cats": "Campus Cats (Outdoor, Direct Sunlight, Ground Shadows, Partial Occlusion)",
        "cycle": "Bicycle (Hostel Corridor, Interior Lighting, Thin Frame & Wheel Spoke Geometry)",
        "flower": "Plumeria Flower (Outdoor Daylight, Handheld, High Contrast Petals)",
        "shoes": "Sneakers Pair (Indoor Floor, Tiled Ground, Cluttered Shadows)",
        "watch": "Wristwatch (Indoor Close-up, Patterned Cloth Background)"
    }
    
    fig, axes = plt.subplots(len(captured_items), 3, figsize=(12, 3.8 * len(captured_items)))
    
    for row_idx, item in enumerate(captured_items):
        img_p = f"captured/images_256/{item}.png"
        mask_p = f"captured/masks_256/{item}.png"
        
        img_bgr = cv2.imread(img_p)
        true_mask_gray = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        true_01 = (true_mask_gray >= 128).astype(np.uint8)
        
        pred_01, _ = run_pipeline(img_bgr, model)
        
        # Overlays: True in Green (0, 255, 0), Predicted in Cyan/Blue (255, 128, 0)
        true_overlay = make_overlay(img_bgr, true_01, color=(0, 255, 0), alpha=0.45)
        pred_overlay = make_overlay(img_bgr, pred_01, color=(0, 165, 255), alpha=0.45) # Orange-cyan
        
        # Save individual 3-panel figure
        item_fig, item_axes = plt.subplots(1, 3, figsize=(12, 4))
        item_axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        item_axes[0].set_title("Input Photo (256x256)")
        item_axes[0].axis('off')
        
        item_axes[1].imshow(cv2.cvtColor(true_overlay, cv2.COLOR_BGR2RGB))
        item_axes[1].set_title("Ground Truth Mask Overlay")
        item_axes[1].axis('off')
        
        item_axes[2].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        item_axes[2].set_title("Predicted Mask Overlay")
        item_axes[2].axis('off')
        
        item_fig.suptitle(descriptions[item], fontsize=12, fontweight='bold')
        item_fig.tight_layout()
        item_fig.savefig(f"{out_dir}/{item}_comparison.png", dpi=250)
        plt.close(item_fig)
        
        # Add to composite figure
        axes[row_idx, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[row_idx, 0].set_title(f"{item.upper()}: Input Photo")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(cv2.cvtColor(true_overlay, cv2.COLOR_BGR2RGB))
        axes[row_idx, 1].set_title(f"{item.upper()}: Ground Truth")
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
        axes[row_idx, 2].set_title(f"{item.upper()}: Prediction")
        axes[row_idx, 2].axis('off')
        
    fig.tight_layout()
    fig.savefig(f"{out_dir}/captured_all_overlays.png", dpi=300)
    plt.close(fig)
    print(f"Generated all captured photo comparison figures in {out_dir}/")

if __name__ == "__main__":
    generate_captured_overlays()
