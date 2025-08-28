import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_fscore_support, accuracy_score, roc_auc_score
import csv
import os

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return np.array(x)
    return np.array(x)

def compute_metrics(outputs, labels):
    if isinstance(outputs, (list, tuple)):
        outputs = torch.cat(outputs, dim=0)
    if isinstance(labels, (list, tuple)):
        labels = torch.cat(labels, dim=0)

    probs = torch.softmax(outputs, dim=1)[:, 1]  # P(spoof)
    preds = torch.argmax(outputs, dim=1)

    labels_np = labels.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    # ROC/EER
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0
    threshold = thresholds[eer_threshold_idx]

    # APCER (False Acceptance), BPCER (False Rejection)
    spoof_idx = labels_np == 1
    real_idx = labels_np == 0
    apcer = np.mean(preds_np[spoof_idx] == 0) if np.any(spoof_idx) else 0.0
    bpcer = np.mean(preds_np[real_idx] == 1) if np.any(real_idx) else 0.0
    acer = (apcer + bpcer) / 2.0
    hter = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0

    # Accuracy and classification metrics
    val_acc = accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except Exception:
        auc = 0.0

    return {
        'val_acc': val_acc,
        'eer': eer,
        'threshold': float(threshold),
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'acer': float(acer),
        'hter': float(hter),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }

def compute_metrics_from_preds(labels, preds, probs=None):
    labels_np = _to_numpy(labels).astype(int).flatten()
    preds_np = _to_numpy(preds).astype(int).flatten()
    probs_np = _to_numpy(probs).astype(float).flatten() if probs is not None else None

    # ensure binary arrays
    if probs_np is None:
        # create simplistic score from preds (0/1) to allow ROC but ROC will be poor
        probs_np = preds_np.copy().astype(float)

    # ROC/EER
    try:
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2.0
        threshold = thresholds[eer_threshold_idx]
    except Exception:
        eer = 0.0
        threshold = 0.0
        fpr = tpr = thresholds = np.array([0.0])

    spoof_idx = labels_np == 1
    real_idx = labels_np == 0
    apcer = np.mean(preds_np[spoof_idx] == 0) if np.any(spoof_idx) else 0.0
    bpcer = np.mean(preds_np[real_idx] == 1) if np.any(real_idx) else 0.0
    acer = (apcer + bpcer) / 2.0
    hter = (fpr[eer_threshold_idx] + (1 - tpr)[eer_threshold_idx]) / 2.0

    val_acc = accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except Exception:
        auc = 0.0

    return {
        'val_acc': float(val_acc),
        'eer': float(eer),
        'threshold': float(threshold),
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'acer': float(acer),
        'hter': float(hter),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }

def _safe_round(x, ndigits=4):
    return '' if x is None else round(x, ndigits)

def log_metrics_to_csv(log_file, epoch, train_loss, train_acc, val_loss, metrics):
    """
    Append a row with epoch/train/val metrics to CSV. Accepts metrics dict
    that may contain None values (will be written as empty cells).
    Writes all metrics: val_acc, apcer, bpcer, acer, hter, eer, threshold, precision, recall, f1_score, auc
    """
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    header = ['epoch','train_loss','train_acc','val_loss','val_acc','thr_acc','balanced_acc',
          'apcer','bpcer','acer','hter','eer','threshold','precision','recall','f1_score','auc']

    row = [
        epoch,
        _safe_round(train_loss),
        _safe_round(train_acc),
        _safe_round(val_loss),
        _safe_round(metrics.get('val_acc')),
        _safe_round(metrics.get('thr_acc')),
        _safe_round(metrics.get('balanced_acc')),        
        _safe_round(metrics.get('apcer')),
        _safe_round(metrics.get('bpcer')),
        _safe_round(metrics.get('acer')),
        _safe_round(metrics.get('hter')),
        _safe_round(metrics.get('eer')),
        _safe_round(metrics.get('threshold')),
        _safe_round(metrics.get('precision')),
        _safe_round(metrics.get('recall')),
        _safe_round(metrics.get('f1_score')),
        _safe_round(metrics.get('auc')),
    ]

    write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)