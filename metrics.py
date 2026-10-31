import numpy as np
import cv2

def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary pixels for a binary mask (H, W) with values {0, 1}.
    A boundary pixel is an object pixel (1) with at least one background pixel (0)
    directly 4-connected (up, down, left, right). Pixels outside the image count as background.
    """
    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"
    h, w = mask.shape
    obj = (mask > 0)
    
    # Pad mask with 0 (outside counts as background)
    padded = np.pad(obj, 1, mode='constant', constant_values=False)
    
    up = padded[0:h, 1:w+1]
    down = padded[2:h+2, 1:w+1]
    left = padded[1:h+1, 0:w]
    right = padded[1:h+1, 2:w+2]
    
    # Has at least one 4-connected neighbor that is 0
    has_bg_neighbor = (~up) | (~down) | (~left) | (~right)
    boundary = obj & has_bg_neighbor
    return boundary

def compute_bf1_single(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Compute Boundary F1 score for a single image pair.
    1. A boundary pixel is an object pixel with a background pixel directly above,
       below, left or right of it. Pixels outside the image count as background.
    2. A predicted boundary pixel is correct if a true boundary pixel lies within its 5x5 window.
       A true boundary pixel is found if a predicted boundary pixel lies within its 5x5 window.
    3. Precision P is fraction of predicted boundary pixels that are correct,
       Recall R is fraction of true boundary pixels that are found.
       BF1 = 2PR / (P + R), or 0 if P + R = 0.
    4. If neither mask has boundary pixels, BF1 is 1. If only one does, BF1 is 0.
    Fast check using cv2.filter2D with a 5x5 ones kernel (allowed under rules).
    """
    pred_bnd = extract_boundary(pred_mask)
    true_bnd = extract_boundary(true_mask)
    
    n_pred = int(np.sum(pred_bnd))
    n_true = int(np.sum(true_bnd))
    
    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0
    
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    
    # true_dilated has >0 if any true boundary pixel lies within the 5x5 window
    true_dilated = cv2.filter2D(true_bnd.astype(np.uint8), -1, kernel5, borderType=cv2.BORDER_CONSTANT) > 0
    # pred_dilated has >0 if any pred boundary pixel lies within the 5x5 window
    pred_dilated = cv2.filter2D(pred_bnd.astype(np.uint8), -1, kernel5, borderType=cv2.BORDER_CONSTANT) > 0
    
    correct_pred = int(np.sum(pred_bnd & true_dilated))
    found_true = int(np.sum(true_bnd & pred_dilated))
    
    precision = correct_pred / n_pred
    recall = found_true / n_true
    
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))

class ConfusionMatrixMeter:
    """
    Accumulates a 2x2 confusion matrix over an entire dataset:
    Rows: Ground Truth [0: Background, 1: Object]
    Cols: Prediction   [0: Background, 1: Object]
    """
    def __init__(self):
        self.cm = np.zeros((2, 2), dtype=np.int64)
        
    def update(self, pred_mask: np.ndarray, true_mask: np.ndarray):
        pred_flat = (pred_mask.flatten() > 0).astype(np.int64)
        true_flat = (true_mask.flatten() > 0).astype(np.int64)
        
        # 0: bg, 1: fg
        # Index: 2 * true + pred
        indices = 2 * true_flat + pred_flat
        counts = np.bincount(indices, minlength=4)
        self.cm[0, 0] += counts[0]  # TN
        self.cm[0, 1] += counts[1]  # FP
        self.cm[1, 0] += counts[2]  # FN
        self.cm[1, 1] += counts[3]  # TP

    def compute_miou(self):
        tn = self.cm[0, 0]
        fp = self.cm[0, 1]
        fn = self.cm[1, 0]
        tp = self.cm[1, 1]
        
        iou_bg = tn / max(1, tn + fp + fn)
        iou_fg = tp / max(1, tp + fp + fn)
        miou = 0.5 * (iou_bg + iou_fg)
        return float(miou), float(iou_bg), float(iou_fg)

def compute_ces_score(miou: float, bf1: float, t_cpu_ms: float, s_disk_bytes: int):
    """
    Computes Q and CES according to the assignment formula:
    Q = 0.70 * mIoU + 0.30 * BF1
    CES = Q * [1 - 0.20 * (T_CPU / 50) - 0.10 * min(1, S_disk / 4 MiB)] if T_CPU <= 50 ms else 0
    where 4 MiB = 4,194,304 bytes.
    """
    q = 0.70 * miou + 0.30 * bf1
    if t_cpu_ms > 50.0:
        ces = 0.0
    else:
        time_penalty = 0.20 * (t_cpu_ms / 50.0)
        size_penalty = 0.10 * min(1.0, s_disk_bytes / 4194304.0)
        penalty_factor = 1.0 - time_penalty - size_penalty
        ces = max(0.0, q * penalty_factor)
    return float(q), float(ces)
