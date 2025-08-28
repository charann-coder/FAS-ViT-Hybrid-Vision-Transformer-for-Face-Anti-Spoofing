import os, time, csv, random, argparse, io, sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from model import FASViTClassifier
from trainer import train_one_epoch_ssl
from dataset_ssl import LCCFASDRotationSSL

# ---------------- Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, inp, tgt):
        logp = -self.ce(inp, tgt)
        p = torch.exp(logp)
        return -((1 - p) ** self.gamma) * logp

# ---------------- CSV utils ----------------
def init_csv(path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'best_acc', 'time_sec'])

def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

# ---------------- Shape check ----------------
def check_pos_embed(model, img_size, patch_size):
    n_tokens_expected = 1 + (img_size // patch_size) ** 2
    try:
        pos = model.feature_extractor.pos
        n_tokens_model = pos.shape[1]
    except Exception:
        # try alternative key name
        try:
            pos = model.feature_extractor.pos_embed
            n_tokens_model = pos.shape[1]
        except Exception:
            print("[SSL INFO] Model has no static pos_embed or it is created dynamically; skipping pos length check.")
            return
    if n_tokens_model != n_tokens_expected:
        raise RuntimeError(
            f"[SSL ERROR] Positional tokens mismatch: model has {n_tokens_model}, "
            f"expected {n_tokens_expected} for img_size={img_size}, patch_size={patch_size}. "
            f"Please ensure your FASViTClassifier uses the same img/patch settings as fine-tune."
        )

def build_parser():
    p = argparse.ArgumentParser()
    # training params
    p.add_argument('--epochs',    type=int,   default=25,    help='Number of epochs')
    p.add_argument('--patience',  type=int,   default=5,     help='Early stopping patience')
    p.add_argument('--batch-size',type=int,   default=8,     help='Batch size')
    p.add_argument('--lr',        type=float, default=1e-4,  help='Learning rate')
    # data / io
    p.add_argument('--data-csv',  type=str,   default='Dataset_csv/ssl_data.csv', help='Path to ssl_data.csv')
    p.add_argument('--data-root', type=str,   default='LCC_dataset/LCC_FASD', help='Root folder for images')
    p.add_argument('--save-dir',  type=str,   default='ssl_checkpoint', help='Checkpoint directory')
    # architecture compatibility knobs (must match fine-tune)
    p.add_argument('--img-size',  type=int,   default=128,   help='Input size (square)')
    p.add_argument('--patch-size',type=int,   default=4,     help='Patch size to MATCH fine-tune model')
    p.add_argument('--depth',     type=int,   default=1,     help='Transformer depth')
    p.add_argument('--embed-dim', type=int,   default=192,   help='Embedding dimension (if supported)')
    return p

def main():
    parser = build_parser()

    # Auto defaults if no CLI args were given (makes `python train_ssl.py` automated)
    if len(sys.argv) == 1:
        args = parser.parse_args([
            '--epochs','25','--patience','5','--batch-size','8','--lr','1e-4',
            '--data-csv','Dataset_csv/ssl_data.csv','--data-root','LCC_dataset/LCC_FASD',
            '--save-dir','ssl_checkpoint','--img-size','128','--patch-size','4',
            '--depth','1','--embed-dim','192'
        ])
    else:
        args = parser.parse_args()

    # ---- Repro ----
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, 'metrics_ssl.csv')

    # ---- Dataset / Loader ----
    ssl_dim = 4  # rotation task (0,90,180,270)
    tf_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor()
    ])
    ds = LCCFASDRotationSSL(args.data_csv, args.data_root, tf_train, real_only=False)
    print(f"🔁 SSL on {len(ds)} samples")

    # check counts for rotation labels
    labels = ds.labels
    counts = [labels.count(i) for i in range(ssl_dim)]
    zero_classes = [i for i, c in enumerate(counts) if c == 0]
    if zero_classes:
        raise ValueError(f"Found zero-sample rotation classes: {zero_classes}. Fix {args.data_csv}.")

    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
    sample_wts = [weights[l].item() for l in labels]
    sampler = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

    num_workers = min(8, (os.cpu_count() or 4))  # safer default on Windows
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # ---- Model (match fine-tune architecture!) ----
    model = FASViTClassifier(
        is_ssl=True,
        num_classes=ssl_dim,
        patch_size=args.patch_size,
        depth=args.depth,
        # uncomment if your FASViTClassifier supports these:
        # img_size=args.img_size, embed_dim=args.embed_dim
    ).to(device)

    # Validate positional token length to guarantee compatibility
    try:
        check_pos_embed(model, args.img_size, args.patch_size)
        print(f"✅ Positional tokens OK for img={args.img_size}, patch={args.patch_size}")
    except RuntimeError as e:
        print(str(e))
        return

    # ---- Optimizer / Scheduler / Loss ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = min(5, max(1, args.epochs // 5))
    sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup))
    crit = FocalLoss(weight=weights.to(device))

    # ---- Train loop ----
    init_csv(metrics_path)
    best_acc = 0.0
    epochs_no_improve = 0

    for ep in trange(1, args.epochs + 1, desc='SSL Epochs'):
        # warmup LR
        if ep <= warmup:
            lr_now = args.lr * ep / warmup
            for g in opt.param_groups:
                g['lr'] = lr_now
        else:
            sched.step()

        loss, acc, secs, _ = train_one_epoch_ssl(model, loader, opt, crit, device, None, ep)

        # Save epoch checkpoint
        ep_path = os.path.join(args.save_dir, f'epoch_{ep}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': ep,
            'acc': acc
        }, ep_path)

        # Save best
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            best_path = os.path.join(args.save_dir, f'best_epoch{ep}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': ep,
                'best_acc': best_acc
            }, best_path)
            with open(os.path.join(args.save_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(ep))
        else:
            epochs_no_improve += 1

        append_csv(metrics_path, [ep, loss, acc, best_acc, secs])
        print(f"Epoch {ep}: loss={loss:.4f}, acc={acc:.4f}, best={best_acc:.4f}")

        if epochs_no_improve >= args.patience:
            print(f"⏹️ Early stopping after {ep} epochs. Best acc: {best_acc:.4f}")
            break

if __name__ == '__main__':
    main()
