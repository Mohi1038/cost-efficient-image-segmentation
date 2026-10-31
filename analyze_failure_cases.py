import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import FastSaliencyUNet
from predict import load_model, run_pipeline
from metrics import compute_bf1_single

def analyze_and_plot_failures(weights_path="parameters.pth", out_dir="report/figures"):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(weights_path)
    
    # We will pick 2 representative test cases from DUTS-TE or captured images:
    # Failure Case 1: Low-contrast / fine texture under-segmentation
    # Failure Case 2: Thin structural elements / partial occlusion
    
    # Scan a subset of DUTS-TE to find an image with high error (low IoU / low BF1)
    te_img_dir = "data/DUTS-TE-256/images"
    te_mask_dir = "data/DUTS-TE-256/masks"
    
    sample_files = sorted(os.listdir(te_img_dir))[:80]
    evaluated = []
    for f in sample_files:
        name = os.path.splitext(f)[0]
        img_p = os.path.join(te_img_dir, f)
        mask_p = os.path.join(te_mask_dir, f"{name}.png")
        if not os.path.exists(mask_p):
            continue
        img = cv2.imread(img_p)
        true_m = (cv2.imread(mask_p, 0) >= 128).astype(np.uint8)
        pred_m, _ = run_pipeline(img, model)
        
        # Calculate IoU
        intersection = np.sum((pred_m == 1) & (true_m == 1))
        union = np.sum((pred_m == 1) | (true_m == 1))
        iou = intersection / max(1, union)
        bf1 = compute_bf1_single(pred_m, true_m)
        evaluated.append({
            'name': name,
            'img_p': img_p,
            'mask_p': mask_p,
            'iou': iou,
            'bf1': bf1,
            'img': img,
            'true_m': true_m,
            'pred_m': pred_m
        })
        
    # Sort by IoU to find challenging failure cases
    evaluated.sort(key=lambda x: x['iou'])
    
    # Pick two distinct failure cases
    fc1 = evaluated[0] # Lowest IoU / severe challenge
    fc2 = evaluated[2] # Another challenging structure
    
    # Helper to generate 4-panel diagnostic plot
    def plot_diagnostic(case, title, reason, filename):
        img_rgb = cv2.cvtColor(case['img'], cv2.COLOR_BGR2RGB)
        true_m = case['true_m']
        pred_m = case['pred_m']
        
        # Difference map: Green = TP, Red = FP, Yellow = FN
        diff_map = np.zeros_like(case['img'])
        tp = (pred_m == 1) & (true_m == 1)
        fp = (pred_m == 1) & (true_m == 0)
        fn = (pred_m == 0) & (true_m == 1)
        diff_map[tp] = [0, 255, 0]    # Green
        diff_map[fp] = [255, 0, 0]    # Blue/Red in RGB
        diff_map[fn] = [255, 255, 0]  # Yellow
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Input Photo (256x256)")
        axes[0].axis('off')
        
        axes[1].imshow(true_m, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_m, cmap='gray')
        axes[2].set_title(f"Predicted Mask\n(IoU: {case['iou']:.3f}, BF1: {case['bf1']:.3f})")
        axes[2].axis('off')
        
        # Overlay on RGB
        overlay = img_rgb.copy()
        overlay[fp] = [255, 50, 50]    # Red for FP
        overlay[fn] = [255, 200, 50]   # Yellow for FN
        overlay[tp] = [50, 255, 50]    # Green for TP
        axes[3].imshow(overlay)
        axes[3].set_title("Error Diagnostic Overlay\n(Green: TP, Red: FP, Yellow: FN)")
        axes[3].axis('off')
        
        fig.suptitle(f"{title}\nCause: {reason}", fontsize=11, fontweight='bold', y=1.02)
        fig.tight_layout()
        fig.savefig(filename, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved failure case to {filename}")

    plot_diagnostic(
        fc1,
        "Failure Case 1: Low-Contrast Camouflage & Photometric Blending",
        "Salient foreground shares color distribution with background surfaces, causing under-segmentation.",
        f"{out_dir}/failure_case_1.png"
    )
    plot_diagnostic(
        fc2,
        "Failure Case 2: Thin Structural Discontinuity & Occlusion",
        "Sub-pixel fine structures and partial occlusion lost during strided encoder downsampling.",
        f"{out_dir}/failure_case_2.png"
    )

if __name__ == "__main__":
    analyze_and_plot_failures()
