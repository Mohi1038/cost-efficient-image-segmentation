import os
import sys

# Critical configuration: Single-threaded CPU execution constraint
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import argparse
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

cv2.setNumThreads(1)
torch.set_num_threads(1)

from model import FastSaliencyUNet
from dataset import get_duts_splits, get_test_dataset, SaliencyDataset
from metrics import ConfusionMatrixMeter, compute_bf1_single, compute_ces_score
from predict import run_pipeline

# Normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_model(weights_path="parameters.pth"):
    model = FastSaliencyUNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    float_state_dict = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in state_dict.items()
    }
    model.load_state_dict(float_state_dict)
    model.eval()
    return model


def evaluate_dataset(model, dataset, name="Dataset", measure_timing=False):
    """
    Evaluates mIoU, BF1, and optionally single-threaded CPU processing time on a dataset.
    """
    meter = ConfusionMatrixMeter()
    bf1_scores = []
    cpu_times_ms = []

    # Warmup once as per rules
    sample_img, _ = dataset[0]
    sample_bgr = (sample_img.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    sample_bgr = np.clip(sample_bgr, 0, 255).astype(np.uint8)
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 256, 256))

    print(f"Evaluating {name} ({len(dataset)} images)...")
    for idx in tqdm(range(len(dataset)), desc=name):
        img_tensor, mask_tensor = dataset[idx]
        true_mask = (mask_tensor[0].numpy() >= 0.5).astype(np.uint8)

        # Evaluate the exact same pipeline used by predict.py. This keeps
        # metrics and timing consistent with the submitted inference script.
        photo_bgr = ((img_tensor.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0).astype(np.uint8)
        pred_mask, elapsed_ms = run_pipeline(photo_bgr, model)
        if measure_timing:
            cpu_times_ms.append(elapsed_ms)

        meter.update(pred_mask, true_mask)
        bf1 = compute_bf1_single(pred_mask, true_mask)
        bf1_scores.append(bf1)

    miou, iou_bg, iou_fg = meter.compute_miou()
    mean_bf1 = float(np.mean(bf1_scores))
    q_score = 0.70 * miou + 0.30 * mean_bf1
    mean_time_ms = float(np.mean(cpu_times_ms)) if cpu_times_ms else 0.0

    return {
        'name': name,
        'count': len(dataset),
        'mIoU': miou,
        'IoU_fg': iou_fg,
        'IoU_bg': iou_bg,
        'BF1': mean_bf1,
        'Q': q_score,
        'T_CPU_ms': mean_time_ms
    }


def get_captured_dataset():
    """
    Loads custom captured phone photos and masks from captured/ directory.
    """
    img_dir = "captured/images_256"
    mask_dir = "captured/masks_256"
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    pairs = []
    for img_p in img_files:
        name = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(mask_dir, f"{name}.png")
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
    if not pairs:
        print(f"Warning: No valid captured pairs found in {img_dir} and {mask_dir}")
        return None
    return SaliencyDataset(pairs, is_train=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation pipeline")
    parser.add_argument("--weights", type=str, default="parameters.pth", help="Path to weights file")
    parser.add_argument("--skip-timing", action="store_true", help="Skip CPU timing")
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.exists(weights_path):
        if os.path.exists("checkpoints/best_full.pth"):
            weights_path = "checkpoints/best_full.pth"
        else:
            print(f"Error: Weights {weights_path} not found.")
            sys.exit(1)

    s_disk_bytes = os.path.getsize(weights_path)
    print(f"Loaded weights from {weights_path} (Size: {s_disk_bytes:,} bytes / {s_disk_bytes / 4194304:.3f} MiB)")

    model = load_model(weights_path)

    # 1. Validation split
    _, val_dataset = get_duts_splits(base_dir="data/DUTS-TR-256", val_ratio=0.10, seed=42)
    val_res = evaluate_dataset(model, val_dataset, name="Validation Set (DUTS-TR split)")

    # 2. DUTS-TE Test set (Timed)
    test_dataset = get_test_dataset(base_dir="data/DUTS-TE-256")
    test_res = evaluate_dataset(model, test_dataset, name="DUTS-TE (5,019 images)", measure_timing=not args.skip_timing)

    # Compute CES on DUTS-TE
    q_test = test_res['Q']
    t_cpu_ms = test_res['T_CPU_ms']
    _, ces_score = compute_ces_score(test_res['mIoU'], test_res['BF1'], t_cpu_ms, s_disk_bytes)
    test_res['CES'] = ces_score

    # 3. Captured photos (if available)
    captured_dataset = get_captured_dataset()
    if captured_dataset is not None:
        captured_res = evaluate_dataset(model, captured_dataset, name="Captured Photos")
    else:
        captured_res = None

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Split / Dataset':<25} | {'mIoU':<8} | {'BF1':<8} | {'Q Score':<8} | {'T_CPU (ms)':<10} | {'S_disk (B)':<10} | {'CES':<8}")
    print("-" * 90)
    print(f"{val_res['name']:<25} | {val_res['mIoU']:<8.4f} | {val_res['BF1']:<8.4f} | {val_res['Q']:<8.4f} | {'-':<10} | {s_disk_bytes:<10} | {'-':<8}")
    print(f"{test_res['name']:<25} | {test_res['mIoU']:<8.4f} | {test_res['BF1']:<8.4f} | {test_res['Q']:<8.4f} | {t_cpu_ms:<10.2f} | {s_disk_bytes:<10} | {ces_score:<8.4f}")
    if captured_res:
        print(f"{captured_res['name']:<25} | {captured_res['mIoU']:<8.4f} | {captured_res['BF1']:<8.4f} | {captured_res['Q']:<8.4f} | {'-':<10} | {s_disk_bytes:<10} | {'-':<8}")
    print("=" * 90)


if __name__ == "__main__":
    main()
