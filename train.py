import os, time, csv, math, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import FASViTClassifier
from dataset import LCCFASDFromCSV
from loss import get_multitask_loss, get_loss
from metrics import compute_metrics_from_preds
import numpy as np
import pandas as pd
from trainer import train_one_epoch, validate

# --- Configuration ---
profile = os.environ.get("TRAIN_PROFILE", "speed")
EPOCHS = 25
PATIENCE = 8
WARMUP_EPOCHS = 2
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = False
USE_SAMPLER = True
LOAD_SSL = True
SSL_CKPT = "ssl_checkpoint/best_epoch7.pth"
RESUME_CKPT  = None

if profile == "accuracy":
    PATCH_SIZE, DEPTH, BACKBONE_LR, HEAD_LR = 2, 2, 3e-5, 1e-4
else: # speed
    PATCH_SIZE, DEPTH, BACKBONE_LR, HEAD_LR = 4, 1, 2e-5, 5e-4

MULTITASK = True
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_NORMALIZE = True
GRADUAL_UNFREEZE = {3: ['xp'], 6: ['fe.fuse']}
MONITOR_METRIC = 'val_acc'

# --- Utility Functions ---
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _safe_round(x):
    return round(float(x), 6) if x is not None else ''

def log_metrics_to_csv(log_file, epoch, train_loss, train_acc, val_loss, val_acc, metrics):
    header = ['epoch','train_loss','train_acc','val_loss','val_acc','apcer','bpcer','acer','hter','eer','auc']
    metrics['val_acc'] = val_acc
    row = [epoch, _safe_round(train_loss), _safe_round(train_acc), _safe_round(val_loss)] + [_safe_round(metrics.get(k)) for k in header[4:]]
    write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    with open(log_file, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

def make_balanced_sampler_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    label_map = {"real":0, "spoof":1}
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(label_map)
    df = df[df['label'].notna()]
    counts = df['label'].value_counts().sort_index()
    inv = 1.0 / counts
    weights = df['label'].map(inv).astype(float).tolist()
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def main():
    set_seed(42)
    train_csv, val_csv, root = "Dataset_csv/train_data.csv", "Dataset_csv/val_data.csv", "LCC_dataset/LCC_FASD"
    norm = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)) if USE_NORMALIZE else nn.Identity()
    train_tf = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(6),
                                   transforms.ColorJitter(0.12,0.12,0.12,0.05), transforms.RandomAffine(0, translate=(0.03,0.03)),
                                   transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), norm])

    train_ds = LCCFASDFromCSV(train_csv, root, train_tf, multitask=MULTITASK)
    val_ds = LCCFASDFromCSV(val_csv, root, val_tf, multitask=MULTITASK)

    sampler = make_balanced_sampler_from_csv(train_csv) if USE_SAMPLER else None
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, shuffle=(sampler is None), num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY, generator=g)

    model = FASViTClassifier(num_classes=NUM_CLASSES, num_spoof_types=len(train_ds.spoof_type_map),
                             patch_size=PATCH_SIZE, depth=DEPTH, is_ssl=False).to(DEVICE)

    if LOAD_SSL and os.path.exists(SSL_CKPT):
        state = torch.load(SSL_CKPT, map_location='cpu').get('model_state_dict', {})
        filtered = {k: v for k, v in state.items() if not any(k.startswith(p) for p in ['head.','spoof_type_head.','ssl_head.'])}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[SSL] Loaded from {SSL_CKPT}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("head") or n.startswith("spoof_type_head")
    print(f"[INFO] total params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.4f}M")

    criterion = get_multitask_loss(spoof_weight=0.3, label_smoothing=0.05, warmup_epochs=WARMUP_EPOCHS) if MULTITASK else get_loss('ce', label_smoothing=0.05)
    
    start_epoch, best_metric, no_improve = 0, -1.0, 0
    writer = SummaryWriter(log_dir=os.path.join("runs", time.strftime("%Y%m%d_%H%M%S")))
    log_file = "training_log.csv"
    checkpoint_dir = "checkpoints"

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if hasattr(criterion, 'epoch'): criterion.epoch = epoch
        print(f"\n--- Epoch {epoch}/{EPOCHS} profile={profile} (patch={PATCH_SIZE} depth={DEPTH}) ---")

        if epoch in GRADUAL_UNFREEZE:
            tags = GRADUAL_UNFREEZE[epoch]
            for n, p in model.named_parameters():
                if any(t in n for t in tags) and not p.requires_grad: p.requires_grad = True
            print(f"[UNFREEZE] epoch {epoch}: unfroze layers for tags {tags}")
        
        optimizer = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")], "lr": BACKBONE_LR},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")], "lr": HEAD_LR}
        ], weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, MULTITASK)
        
        val_loss, val_acc, metrics = validate(model, val_loader, criterion, DEVICE, MULTITASK)
        
        monitor_metric = val_acc
        scheduler.step(monitor_metric)

        
        os.makedirs(checkpoint_dir, exist_ok=True)

        is_best = monitor_metric > best_metric
        if is_best:
            best_metric = monitor_metric
            no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, "best.pth"))
            print(f"✅ New best model saved with Val Acc: {best_metric:.4f}")
        else:
            no_improve += 1

        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, epoch_checkpoint_path)
        print(f"✅ Checkpoint for epoch {epoch} saved to {epoch_checkpoint_path}")
        
        # --- Logging ---
        log_metrics_to_csv(log_file, epoch, train_loss, train_acc, val_loss, val_acc, metrics)
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("EER", metrics.get('eer', 0), epoch)

        print(f"[EPOCH {epoch}] train_acc={train_acc:.4f} monitor({MONITOR_METRIC})={monitor_metric:.4f} best={best_metric:.4f} EER={metrics.get('eer','-')}")
        
        if no_improve >= PATIENCE:
            print(f"[EARLY STOP] No improvement in {PATIENCE} epochs, stopping training.")
            break
            
    writer.close()
    print(f"Training finished. Best Val Acc: {best_metric:.4f}")

if __name__ == "__main__":
    main()