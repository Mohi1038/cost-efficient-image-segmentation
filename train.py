import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FastSaliencyUNet
from dataset import get_duts_splits
from metrics import ConfusionMatrixMeter, compute_bf1_single

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def evaluate_validation(model, val_loader, device):
    model.eval()
    meter = ConfusionMatrixMeter()
    bf1_scores = []
    total_loss = 0.0
    criterion = BCEDiceLoss()
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.uint8).cpu().numpy()
            trues = (masks >= 0.5).to(torch.uint8).cpu().numpy()
            
            for b in range(preds.shape[0]):
                p_mask = preds[b, 0]
                t_mask = trues[b, 0]
                meter.update(p_mask, t_mask)
                bf1 = compute_bf1_single(p_mask, t_mask)
                bf1_scores.append(bf1)
                
    mean_loss = total_loss / len(val_loader.dataset)
    miou, iou_bg, iou_fg = meter.compute_miou()
    mean_bf1 = float(np.mean(bf1_scores))
    q_score = 0.70 * miou + 0.30 * mean_bf1
    return mean_loss, miou, mean_bf1, q_score, iou_bg, iou_fg


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")
    
    train_dataset, val_dataset = get_duts_splits(
        base_dir=args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )
    
    model = FastSaliencyUNet().to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"Initialized FastSaliencyUNet with {params_count:,} parameters.")
    
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_q = -1.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_bf1': [],
        'val_q': []
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\nStarting Training...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_t0 = time.time()
        
        for step, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                cur_loss = running_loss / ((step + 1) * args.batch_size)
                print(f"  [Epoch {epoch:02d}] Step {step+1:03d}/{len(train_loader):03d} | Loss: {cur_loss:.4f}", flush=True)
            
        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        
        # Validation
        val_loss, val_miou, val_bf1, val_q, iou_bg, iou_fg = evaluate_validation(model, val_loader, device)
        epoch_dur = time.time() - epoch_t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_bf1'].append(val_bf1)
        history['val_q'].append(val_q)
        
        print(f"\n>>> Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"mIoU: {val_miou:.4f} (fg: {iou_fg:.4f}, bg: {iou_bg:.4f}) | "
              f"BF1: {val_bf1:.4f} | Q: {val_q:.4f}", flush=True)
        
        if val_q > best_q:
            best_q = val_q
            # Save parameters at half-precision to minimize S_disk
            half_state_dict = {k: v.half() for k, v in model.state_dict().items()}
            torch.save(half_state_dict, 'parameters.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_q': val_q,
                'val_miou': val_miou,
                'val_bf1': val_bf1
            }, 'checkpoints/best_full.pth')
            print(f"  --> Saved new best parameters to parameters.pth (Q: {best_q:.4f}, Size: {os.path.getsize('parameters.pth'):,} bytes)\n", flush=True)
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Q: {best_q:.4f}", flush=True)

    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE + Dice)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), history['val_miou'], label='Val mIoU', marker='o')
    plt.plot(range(1, args.epochs + 1), history['val_bf1'], label='Val BF1', marker='^')
    plt.plot(range(1, args.epochs + 1), history['val_q'], label='Val Q', marker='x', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.title('Validation Metrics (mIoU, BF1, Q)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    print("Saved training curves plot to training_curves.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FastSaliencyUNet on DUTS-TR")
    parser.add_argument('--data-dir', type=str, default='data/DUTS-TR-256')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    train(args)
