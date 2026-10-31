import os
import sys

# Critical configuration: Single-threaded CPU execution constraint
# MUST be set before importing numpy, cv2, or torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import argparse
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from model import FastSaliencyUNet

# Pre-computed ImageNet normalization constants as float32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_model(weights_path="parameters.pth"):
    """
    Loads model parameters on CPU. Supports float16 and float32 checkpoints.
    """
    model = FastSaliencyUNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    # If weights are stored in float16, convert floating tensors to float32 for CPU inference
    float_state_dict = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in state_dict.items()
    }
    model.load_state_dict(float_state_dict)
    model.eval()
    return model


def fast_guided_filter(guide_gray: np.ndarray, p: np.ndarray, r: int = 6, eps: float = 1e-3) -> np.ndarray:
    """
    O(1) Fast Guided Filter using pure OpenCV box filters (~0.6 ms).
    Transfers sharp physical boundary edges from the guide image into the probability map.
    """
    guide = guide_gray.astype(np.float32) * (1.0 / 255.0) if guide_gray.dtype == np.uint8 else guide_gray
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(guide, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(guide * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(guide * guide, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    
    q = mean_a * guide + mean_b
    return np.clip(q, 0.0, 1.0)


def postprocess_saliency(prob: np.ndarray, photo_bgr: np.ndarray, th: float = 0.65) -> np.ndarray:
    """
    Fast edge-guided post-processing pipeline (< 1.8 ms total).
    1. Edge-preserving guided filtering with grayscale guide.
    2. Adaptive thresholding.
    3. Morphological opening to sever thin spurious bridges.
    4. Saliency-mass connected component selection.
    5. Clean interior hole filling.
    """
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Guided edge sharpening
    guided_p = fast_guided_filter(gray, prob, r=6, eps=1e-3)
    
    # 2. Binary threshold
    binary = (guided_p >= th).astype(np.uint8)
    
    # 3. Morphological opening (kernel 3x3)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    
    # 4. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
    if num_labels <= 1:
        return opened
        
    cleaned = np.zeros_like(opened)
    h, w = prob.shape
    cy, cx = h / 2.0, w / 2.0
    
    scores = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 100:  # Suppress tiny background noise
            continue
        comp_mask = (labels == lbl)
        mean_sal = prob[comp_mask].mean()
        if mean_sal < 0.45:  # Drop low-confidence blobs
            continue
        cen_x, cen_y = centroids[lbl]
        dist = np.sqrt((cen_x - cx)**2 + (cen_y - cy)**2) / (w * 0.5)
        # sal**2 strongly down-weights medium-confidence regions to reduce over-prediction
        score = area * (mean_sal ** 2.0) / (1.0 + 0.3 * dist)
        scores.append((score, lbl))
        
    if scores:
        scores.sort(reverse=True)
        top_score = scores[0][0]
        for sc, lbl in scores:
            if sc >= 0.20 * top_score:  # Stricter: keep only dominant components
                cleaned[labels == lbl] = 1
    else:
        cleaned = opened
        
    # 5. Fill tiny interior pinholes without filling structural geometric voids (e.g. bike frames)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    return filled


def run_pipeline(photo_256, model):
    """
    The timed pipeline as specified by Section 2b:
    Starts with a 256x256 photo in memory (NumPy BGR array) and ends with its mask,
    a 256x256 array of 0s and 1s, in memory.
    Every step in between counts, including colour conversion, normalization,
    tensor conversion, forward inference, sigmoid, and thresholding.
    """
    t0 = time.process_time()
    
    # 1. Colour conversion: BGR to RGB
    rgb = cv2.cvtColor(photo_256, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    
    # 2. Normalization
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    
    # 3. Tensor layout: HWC -> CHW, add batch dim
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    
    # 4. Network forward pass
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].numpy()
        
    # 5. Edge-guided refinement & binary mask generation {0, 1}
    mask_01 = postprocess_saliency(probs, photo_256)
    
    t1 = time.process_time()
    elapsed_ms = (t1 - t0) * 1000.0
    return mask_01, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Predict saliency mask for a photo.")
    parser.add_argument("--image", "--input", "-i", dest="image", type=str, required=True, help="Path to input photo")
    parser.add_argument("--out", "--output", "-o", dest="out", type=str, required=True, help="Path to save output mask PNG")
    parser.add_argument("--weights", "-w", type=str, default="parameters.pth", help="Path to parameter file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image {args.image} does not exist.", file=sys.stderr)
        sys.exit(1)

    weights_path = args.weights
    if not os.path.exists(weights_path):
        # Fallback to checkpoints/best_full.pth if parameters.pth is not present yet
        if os.path.exists("checkpoints/best_full.pth"):
            weights_path = "checkpoints/best_full.pth"
        else:
            print(f"Error: weights {weights_path} not found.", file=sys.stderr)
            sys.exit(1)

    s_disk_bytes = os.path.getsize(weights_path)
    model = load_model(weights_path)

    # Reading and resizing the original photo (not timed as per Section 2b)
    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        print(f"Error: failed to read image {args.image}", file=sys.stderr)
        sys.exit(1)

    # Section 1b: Resize every photo to 256x256 with cv2.INTER_AREA
    photo_256 = cv2.resize(orig_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    # Warmup run (Section 2b: "Run your pipeline once on any photo before you start timing")
    mask_01, _ = run_pipeline(photo_256, model)

    # Save the predicted mask as 0 and 255 PNG
    mask_255 = mask_01 * 255
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, mask_255)

    # Times 20 more runs on the same photo
    times_ms = []
    for _ in range(20):
        _, dt = run_pipeline(photo_256, model)
        times_ms.append(dt)

    t_cpu_ms = float(np.mean(times_ms))

    # Print exactly two lines as specified
    print(f"T CPU ms: {t_cpu_ms:.2f}")
    print(f"S disk bytes: {s_disk_bytes}")


if __name__ == "__main__":
    main()
