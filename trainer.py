import time
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from metrics import compute_metrics_from_preds

# logger
logger = logging.getLogger('train_debug_logger')
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    fh = logging.FileHandler('train_debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


def train_one_epoch(model, dataloader, optimizer, criterion, device, multitask=True, debug=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc="🧠 Training Epoch", unit="batch", dynamic_ncols=True, leave=True)
    for batch in pbar:
        # [FIX] Correctly unpack the 3-element batch (images, targets, paths)
        images, targets, _ = batch
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        if multitask:
            labels, spoof_types = targets
            labels = labels.to(device)
            spoof_types = spoof_types.to(device)
            loss = criterion(outputs, labels, spoof_types)
            preds = outputs[0].argmax(dim=1)
        else:
            labels = targets.to(device)
            real_spoof_out = outputs[0]
            loss = criterion(real_spoof_out, labels)
            preds = real_spoof_out.argmax(dim=1)

        if torch.isnan(loss).any().item():
            logger.warning("NaN loss encountered; skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs

        pbar.set_postfix({
            "Loss": f"{running_loss / total:.4f}",
            "Acc": f"{correct / total:.4f}",
            "Seen": f"{total}/{len(dataloader.dataset)}",
            "Time": f"{(time.time() - start_time):.1f}s"
        })

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.inference_mode()
def validate(model, dataloader, criterion, device, multitask=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    start_time = time.time()
    pbar = tqdm(dataloader, desc="📊 Validation Epoch", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        # [FIX] Correctly unpack the 3-element batch to match the dataset output
        images, targets, _ = batch
        images = images.to(device)

        outputs = model(images)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        if multitask:
            labels, spoof_types = targets
            labels = labels.to(device)
            spoof_types = spoof_types.to(device)
            loss = criterion(outputs, labels, spoof_types)
        else:
            labels = targets.to(device)
            loss = criterion(outputs[0], labels)

        real_spoof_out = outputs[0]
        probs = F.softmax(real_spoof_out, dim=1) if real_spoof_out.ndim == 2 else real_spoof_out
        preds = real_spoof_out.argmax(dim=1)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        if probs.ndim == 2 and probs.shape[1] > 1:
            all_probs.extend(probs[:, 1].cpu().tolist())
        else:
            all_probs.extend([float(x) for x in probs.flatten().cpu().tolist()])

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    try:
        metrics = compute_metrics_from_preds(all_labels, all_preds, all_probs)
    except Exception:
        logger.exception("compute_metrics_from_preds failed")
        metrics = {"val_acc": acc, "eer": None, "acer": None, "auc": None}

    print(f"✅ Validation done in {time.time() - start_time:.1f}s | 🎯 Val Acc: {acc:.4f} | EER: {metrics.get('eer', 0):.4f} | ACER: {metrics.get('acer', 0):.4f}")
    return avg_loss, acc, metrics