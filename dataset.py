import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Standard ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

class SaliencyDataset(Dataset):
    """
    Dataset for Saliency Segmentation supporting DUTS-TR (train/val splits), DUTS-TE,
    and custom captured photos.
    Images and masks are 256x256.
    """
    def __init__(self, file_pairs, is_train: bool = False):
        self.file_pairs = file_pairs
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]
        
        # Load image (BGR) and convert to RGB
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        
        # If not 256x256, resize with cv2.INTER_AREA as required
        if img_bgr.shape[:2] != (256, 256):
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Load mask
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask_gray.shape[:2] != (256, 256):
            mask_gray = cv2.resize(mask_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Threshold: pixel >= 128 is object
        mask_bin = (mask_gray >= 128).astype(np.float32)
        
        # Training data augmentation
        if self.is_train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
                mask_bin = np.ascontiguousarray(np.fliplr(mask_bin))
                
            # Random brightness and contrast adjustment
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.15, 0.15)  # Contrast
                beta = np.random.uniform(-0.1, 0.1)           # Brightness
                img_rgb = np.clip(img_rgb * alpha + beta, 0.0, 1.0)
                
        # Normalize with ImageNet mean and std
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert to tensors: Image (3, 256, 256), Mask (1, 256, 256)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()
        
        return img_tensor, mask_tensor


def get_duts_splits(base_dir='data/DUTS-TR-256', val_ratio=0.10, seed=42):
    """
    Splits DUTS-TR into train and validation sets with a reproducible seed.
    Returns (train_dataset, val_dataset).
    """
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    
    files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    pairs = []
    for f in files:
        name = os.path.splitext(f)[0]
        img_p = os.path.join(img_dir, f)
        mask_p = os.path.join(mask_dir, f'{name}.png')
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
            
    # Reproducible shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    
    val_count = int(round(len(pairs) * val_ratio))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    
    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    
    print(f"Total DUTS-TR samples: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    train_dataset = SaliencyDataset(train_pairs, is_train=True)
    val_dataset = SaliencyDataset(val_pairs, is_train=False)
    
    return train_dataset, val_dataset


def get_test_dataset(base_dir='data/DUTS-TE-256'):
    """
    Returns DUTS-TE test dataset.
    """
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    
    files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    pairs = []
    for f in files:
        name = os.path.splitext(f)[0]
        img_p = os.path.join(img_dir, f)
        mask_p = os.path.join(mask_dir, f'{name}.png')
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
            
    print(f"Total DUTS-TE samples: {len(pairs)}")
    return SaliencyDataset(pairs, is_train=False)
