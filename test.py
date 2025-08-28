import os, json, csv, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
from dataset import LCCFASDFromCSV
from model import FASViTClassifier

# ----------------- Default Config (can be overridden by CLI) -----------------
DEFAULT_TEST_CSV = "Dataset_csv/test_data.csv"
DEFAULT_IMG_ROOT = "LCC_dataset/LCC_FASD"
DEFAULT_CKPT = "checkpoints/epoch_19.pth"
IMG_SIZE = 128
PATCH_SIZE = 4
DEPTH = 1
NUM_SPOOF_TYPES = 3
IS_MULTITASK = False          # False since test csv has no spoof_type column
NORMALIZE = True
BATCH_SIZE = 32

# ----------------- Transforms -----------------
norm = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)) if NORMALIZE else (lambda x: x)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    norm
])

# ----------------- Metrics -----------------
def compute_apcer_bpcer(labels, preds):
    labels = np.array(labels); preds = np.array(preds)
    real_mask = labels == 0
    spoof_mask = labels == 1
    apcer = ((preds[spoof_mask] == 0).sum() / spoof_mask.sum()) if spoof_mask.any() else 0.0
    bpcer = ((preds[real_mask] == 1).sum() / real_mask.sum()) if real_mask.any() else 0.0
    acer = 0.5 * (apcer + bpcer)
    return apcer, bpcer, acer

def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thr[idx]

def sweep_thresholds(labels, probs, mode="acer", steps=301):
    labels = np.array(labels); probs = np.array(probs)
    best_stats = None
    thresholds = np.linspace(0,1,steps)
    best_score = -1e9
    for t in thresholds:
        preds = (probs >= t).astype(int)
        apcer, bpcer, acer = compute_apcer_bpcer(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        acc = (preds == labels).mean()
        score = -acer if mode == "acer" else bal_acc
        if score > best_score:
            best_score = score
            best_stats = {
                "threshold": t,
                "apcer": apcer,
                "bpcer": bpcer,
                "acer": acer,
                "balanced_acc": bal_acc,
                "acc": acc
            }
    return best_stats["threshold"], best_stats

# ----------------- Plots -----------------
def plot_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return cm

def plot_roc_curve_and_auc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return auc_score

def plot_confidence_hist(confidences, labels_bin, path):
    df = pd.DataFrame({"confidence": confidences, "label": ["spoof" if l==1 else "real" for l in labels_bin]})
    plt.figure()
    sns.histplot(df, x="confidence", hue="label", bins=30, stat="count", kde=True, multiple="stack")
    plt.xlabel("Prob(spoof)"); plt.title("Confidence Histogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ----------------- Evaluation Loop -----------------
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs, all_files = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", ncols=100):
            # dataset returns (img,label,...) flexible
            if len(batch) == 4:
                images, labels, spoof_types, filenames = batch
            else:
                images, labels, filenames = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                spoof_logits = outputs[0]
            else:
                spoof_logits = outputs
            probs = torch.softmax(spoof_logits, dim=1)
            preds = probs.argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
            all_files.extend(filenames)
    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "files": all_files
    }

# ----------------- CLI -----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--root", default=DEFAULT_IMG_ROOT)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out_prefix", default="test")
    ap.add_argument("--mode", choices=["min_acer","max_bal"], default="min_acer",
                    help="Calibrated threshold selection strategy.")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    ap.add_argument("--save-threshold-json", action="store_true")
    ap.add_argument("--override-threshold", type=float, default=None,
                    help="Skip sweep, force this threshold for calibrated metrics.")
    ap.add_argument("--steps", type=int, default=301, help="Threshold sweep steps.")
    return ap.parse_args()

# ----------------- Main -----------------
def main():
    # Parse args first
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Now we can use args
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = f"metrics_{timestamp}.csv" if args.out_prefix == "test" else f"{args.out_prefix}_metrics.csv"

    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"[INFO] Created plots directory: {plots_dir}")

    ds = LCCFASDFromCSV(args.csv, args.root, transform, multitask=IS_MULTITASK)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FASViTClassifier(num_classes=2,
                             num_spoof_types=NUM_SPOOF_TYPES,
                             patch_size=PATCH_SIZE,
                             depth=DEPTH).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[INFO] Loaded checkpoint {args.ckpt} (epoch={ckpt.get('epoch','?')})")

    res = evaluate(model, loader, device)
    labels = res["labels"]; preds_raw = res["preds"]; probs = res["probs"]

    # Raw metrics
    report = classification_report(labels, preds_raw, target_names=["real","spoof"], digits=4)
    print("\nClassification Report (thr=0.5 argmax):\n", report)

    if not args.no_plots:
        plot_path = lambda filename: os.path.join(plots_dir, filename)
        plot_confusion_matrix(labels, preds_raw, ["real","spoof"], plot_path(f"{args.out_prefix}_confusion_matrix_raw.png"))
        plot_confidence_hist(probs, labels, plot_path(f"{args.out_prefix}_confidence_hist.png"))
        auc = plot_roc_curve_and_auc(labels, probs, plot_path(f"{args.out_prefix}_roc_curve.png"))
    else:
        auc = roc_auc_score(labels, probs)

    apcer0, bpcer0, acer0 = compute_apcer_bpcer(labels, preds_raw)
    eer, eer_thr = compute_eer(labels, probs)
    bal_raw = balanced_accuracy_score(labels, preds_raw)
    acc_raw = (preds_raw == labels).mean()
    print(f"\nRAW  Acc:{acc_raw:.4f} BalAcc:{bal_raw:.4f} APCER:{apcer0:.4f} BPCER:{bpcer0:.4f} ACER:{acer0:.4f}")
    print(f"EER:{eer:.4f} @ {eer_thr:.4f}  AUC:{auc:.4f}")

    # Threshold selection
    if args.override_threshold is not None:
        chosen_name = "override"
        thr = args.override_threshold
        print(f"[INFO] Using override threshold {thr:.4f}")
    else:
        stats_acer_thr, stats_acer = sweep_thresholds(labels, probs, mode="acer", steps=args.steps)
        stats_bal_thr, stats_bal = sweep_thresholds(labels, probs, mode="balanced", steps=args.steps)
        print(f"\n[SWEEP] Min ACER thr={stats_acer['threshold']:.4f} "
              f"ACER={stats_acer['acer']:.4f} APCER={stats_acer['apcer']:.4f} "
              f"BPCER={stats_acer['bpcer']:.4f} BalAcc={stats_acer['balanced_acc']:.4f} Acc={stats_acer['acc']:.4f}")
        print(f"[SWEEP] Max BalAcc thr={stats_bal['threshold']:.4f} "
              f"BalAcc={stats_bal['balanced_acc']:.4f} ACER={stats_bal['acer']:.4f} "
              f"APCER={stats_bal['apcer']:.4f} BPCER={stats_bal['bpcer']:.4f} Acc={stats_bal['acc']:.4f}")

        if args.mode == "min_acer":
            chosen_name = "min_acer"; thr = stats_acer['threshold']
        else:
            chosen_name = "max_bal"; thr = stats_bal['threshold']

    preds_calib = (probs >= thr).astype(int)
    apcer_c, bpcer_c, acer_c = compute_apcer_bpcer(labels, preds_calib)
    bal_c = balanced_accuracy_score(labels, preds_calib)
    acc_c = (preds_calib == labels).mean()
    print(f"\nCALIBRATED ({chosen_name}) Thr={thr:.4f} Acc:{acc_c:.4f} BalAcc:{bal_c:.4f} "
          f"ACER:{acer_c:.4f} APCER:{apcer_c:.4f} BPCER:{bpcer_c:.4f}")

    if not args.no_plots:
        plot_confusion_matrix(labels, preds_calib, ["real","spoof"], 
                             plot_path(f"{args.out_prefix}_confusion_matrix_calibrated.png"))

    # Save metrics row
    metrics_csv = "test_metrics_log.csv"
    write_header = (not os.path.exists(metrics_csv)) or os.path.getsize(metrics_csv) == 0
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["mode","threshold","acc_raw","bal_raw","acer_raw","apcer_raw","bpcer_raw",
                        "acc_calib","bal_calib","acer_calib","apcer_calib","bpcer_calib",
                        "auc","eer","eer_thr"])
        w.writerow([chosen_name, thr,
                    acc_raw, bal_raw, acer0, apcer0, bpcer0,
                    acc_c, bal_c, acer_c, apcer_c, bpcer_c,
                    auc, eer, eer_thr])
    print(f"[INFO] Appended metrics -> {metrics_csv}")

    if args.save_threshold_json:
        threshold_json = f"{args.out_prefix}_threshold.json"
        with open(threshold_json,"w") as jf:
            json.dump({
                "mode": chosen_name,
                "threshold": float(thr),
                "raw": {"acc": float(acc_raw), "balanced_acc": float(bal_raw), 
                       "acer": float(acer0), "apcer": float(apcer0), "bpcer": float(bpcer0)},
                "calibrated": {"acc": float(acc_c), "balanced_acc": float(bal_c), 
                              "acer": float(acer_c), "apcer": float(apcer_c), "bpcer": float(bpcer_c)},
                "auc": float(auc), "eer": float(eer), "eer_thr": float(eer_thr)
            }, jf, indent=2)
        print(f"[INFO] Saved threshold JSON -> {threshold_json}")

    # Predictions CSV (raw + calibrated)
    df = pd.DataFrame({
        "filename": res["files"],
        "label": ["spoof" if l==1 else "real" for l in labels],
        "prob_spoof": probs,
        "pred_raw": ["spoof" if p==1 else "real" for p in preds_raw],
        "pred_calibrated": ["spoof" if p==1 else "real" for p in preds_calib]
    })
    pred_csv = f"{args.out_prefix}_predictions.csv"
    df.to_csv(pred_csv, index=False)
    print(f"[INFO] Saved predictions -> {pred_csv}")

    print("\nSample (raw -> calibrated):")
    for i in range(min(5, len(df))):
        print(f"{df.filename[i]:40s} {df.label[i]:5s} {df.pred_raw[i]:5s} -> {df.pred_calibrated[i]:5s} {df.prob_spoof[i]:.4f}")

if __name__ == "__main__":
    main()