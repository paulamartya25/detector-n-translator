"""
evaluate_age_model.py
=====================
Measures age-prediction MAE for:
  A) InsightFace buffalo_l default model
  B) Your custom retrained model (models/age_best.pt)

Both evaluated on the same UTKFace held-out validation split (15%).
Indian vs non-Indian breakdown shown separately.

Run:
    python evaluate_age_model.py

Outputs:
    outputs/evaluation_results.txt   <- paste these numbers into your SoP
    outputs/evaluation_plot.png      <- scatter plot
"""

import os, sys, glob, time
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/UTKFace"
MODEL_PATH   = "models/age_best.pt"
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EVAL_SAMPLES = 500   # use 500 images per group for speed; set None for full set

# ── Check dataset ─────────────────────────────────────────────────────────────
imgs = glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True)
if len(imgs) < 100:
    print("ERROR: UTKFace dataset not found at", DATASET_PATH)
    print("Download it first:")
    print("  kaggle datasets download -d jangedoo/utkface-new -p data --unzip")
    sys.exit(1)

print(f"Found {len(imgs):,} UTKFace images")

# ── Parse filenames (format: AGE_GENDER_RACE_*.jpg) ───────────────────────────
records = []
for p in imgs:
    parts = os.path.basename(p).split("_")
    if len(parts) < 4:
        continue
    try:
        age  = int(parts[0])
        race = int(parts[2])
        if not (1 <= age <= 90):
            continue
        records.append({"path": p, "age": age,
                        "is_indian": (race == 3)})
    except (ValueError, IndexError):
        continue

df = pd.DataFrame(records)
print(f"Valid images: {len(df):,}  |  Indian: {df.is_indian.sum():,} "
      f"({df.is_indian.mean()*100:.1f}%)")

# ── Reproducible validation split (same seed as training) ─────────────────────
_, val_df = train_test_split(df, test_size=0.15, random_state=42)
val_df = val_df.reset_index(drop=True)
print(f"Validation set: {len(val_df):,} images")

# Sub-sample for speed
if MAX_EVAL_SAMPLES and len(val_df) > MAX_EVAL_SAMPLES * 2:
    indian_val     = val_df[val_df.is_indian].sample(
                        min(MAX_EVAL_SAMPLES, val_df.is_indian.sum()),
                        random_state=42)
    non_indian_val = val_df[~val_df.is_indian].sample(
                        MAX_EVAL_SAMPLES, random_state=42)
    val_df = pd.concat([indian_val, non_indian_val]).sample(frac=1, random_state=42)
    print(f"Sub-sampled to {len(val_df):,} images for speed")

# ── Helper: load and preprocess face crop ─────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path, size=224):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,H,W)

# ─────────────────────────────────────────────────────────────────────────────
# PART A — InsightFace default model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A: InsightFace buffalo_l default age model")
print("="*60)

insightface_preds = []
insightface_trues = []
insightface_indian_preds = []
insightface_indian_trues = []

try:
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l",
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(160, 160))  # small det for speed

    t0 = time.time()
    skipped = 0
    for i, row in val_df.iterrows():
        img_bgr = cv2.imread(row.path)
        if img_bgr is None:
            skipped += 1
            continue
        # Resize for faster detection
        img_bgr = cv2.resize(img_bgr, (160, 160))
        faces = app.get(img_bgr)
        if not faces:
            skipped += 1
            continue
        pred_age = float(faces[0].age)
        insightface_preds.append(pred_age)
        insightface_trues.append(row.age)
        if row.is_indian:
            insightface_indian_preds.append(pred_age)
            insightface_indian_trues.append(row.age)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(val_df) - i - 1)
            print(f"  [{i+1}/{len(val_df)}]  elapsed {elapsed:.0f}s  "
                  f"ETA {remaining:.0f}s", end="\r")

    print()
    if insightface_preds:
        if_mae_all    = mean_absolute_error(insightface_trues, insightface_preds)
        if_bias_all   = np.mean(np.array(insightface_preds) - np.array(insightface_trues))
        if_mae_indian = mean_absolute_error(insightface_indian_trues,
                                            insightface_indian_preds) if insightface_indian_preds else float("nan")
        if_bias_indian = np.mean(np.array(insightface_indian_preds) -
                                 np.array(insightface_indian_trues)) if insightface_indian_preds else float("nan")
        print(f"  InsightFace — Overall  MAE: {if_mae_all:.2f}y   Bias: {if_bias_all:+.2f}y")
        print(f"  InsightFace — Indian   MAE: {if_mae_indian:.2f}y   Bias: {if_bias_indian:+.2f}y")
        print(f"  (skipped {skipped} images with no face detected)")
    else:
        print("  No predictions made — check InsightFace installation")
        if_mae_all = if_mae_indian = float("nan")

except ImportError:
    print("  InsightFace not installed — skipping Part A")
    if_mae_all = if_mae_indian = float("nan")
    insightface_preds = []

# ─────────────────────────────────────────────────────────────────────────────
# PART B — Your custom retrained model (models/age_best.pt)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART B: Custom retrained model (models/age_best.pt)")
print("="*60)

custom_preds        = []
custom_trues        = []
custom_indian_preds = []
custom_indian_trues = []

if not os.path.exists(MODEL_PATH):
    print(f"  Model not found at {MODEL_PATH}")
    print("  Train it first:  python kaggle_training/train_local.py")
    custom_mae_all = custom_mae_indian = float("nan")
else:
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        # Detect v1 vs v2 from file size
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        is_v2   = size_mb > 30
        print(f"  Model size: {size_mb:.1f} MB  →  {'v2 (EfficientNetB3)' if is_v2 else 'v1 (MobileNetV2)'}")

        class _AgeNet(nn.Module):
            def __init__(self, v2=False):
                super().__init__()
                if v2:
                    base = tv_models.efficientnet_b3(weights=None)
                    self.features = base.features
                    self.pool     = nn.AdaptiveAvgPool2d(1)
                    self.head     = nn.Sequential(
                        nn.Dropout(0.4), nn.Linear(1536, 512),
                        nn.BatchNorm1d(512), nn.GELU(),
                        nn.Dropout(0.3), nn.Linear(512, 128),
                        nn.GELU(), nn.Linear(128, 1),
                    )
                else:
                    base = tv_models.mobilenet_v2(weights=None)
                    self.features = base.features
                    self.pool     = nn.AdaptiveAvgPool2d(1)
                    self.head     = nn.Sequential(
                        nn.Dropout(0.3), nn.Linear(1280, 512),
                        nn.BatchNorm1d(512), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(512, 128),
                        nn.ReLU(), nn.Linear(128, 1),
                    )
            def forward(self, x):
                x = self.pool(self.features(x)).flatten(1)
                return self.head(x).squeeze(1)

        net = _AgeNet(v2=is_v2)
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                       weights_only=True))
        net.eval()
        print("  Model loaded ✅")

        input_size = 260 if is_v2 else 224
        t0 = time.time()

        with torch.no_grad():
            for i, row in val_df.iterrows():
                inp = preprocess(row.path, size=input_size)
                if inp is None:
                    continue
                pred = float(net(torch.from_numpy(inp)).item())
                pred = max(1, min(90, pred))

                custom_preds.append(pred)
                custom_trues.append(row.age)
                if row.is_indian:
                    custom_indian_preds.append(pred)
                    custom_indian_trues.append(row.age)

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    remaining = elapsed / (i+1) * (len(val_df) - i - 1)
                    print(f"  [{i+1}/{len(val_df)}]  elapsed {elapsed:.0f}s  "
                          f"ETA {remaining:.0f}s", end="\r")

        print()
        custom_mae_all    = mean_absolute_error(custom_trues, custom_preds)
        custom_bias_all   = np.mean(np.array(custom_preds) - np.array(custom_trues))
        custom_mae_indian = mean_absolute_error(custom_indian_trues,
                                                custom_indian_preds) if custom_indian_preds else float("nan")
        custom_bias_indian = np.mean(np.array(custom_indian_preds) -
                                     np.array(custom_indian_trues)) if custom_indian_preds else float("nan")

        print(f"  Custom model  — Overall  MAE: {custom_mae_all:.2f}y   Bias: {custom_bias_all:+.2f}y")
        print(f"  Custom model  — Indian   MAE: {custom_mae_indian:.2f}y   Bias: {custom_bias_indian:+.2f}y")

    except Exception as e:
        print(f"  Failed to load custom model: {e}")
        custom_mae_all = custom_mae_indian = float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY + SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

lines = [
    "Age Model Evaluation — Detector N Translator",
    f"Validation set: {len(val_df):,} images from UTKFace (15% held-out)",
    f"Indian subset:  {val_df.is_indian.sum():,} images",
    "",
    f"{'Model':<30} {'Overall MAE':>12} {'Indian MAE':>12}",
    "-" * 56,
    f"{'InsightFace buffalo_l (default)':<30} {if_mae_all:>11.2f}y {if_mae_indian:>11.2f}y",
    f"{'Custom retrained (MobileNetV2/EB3)':<30} {custom_mae_all:>11.2f}y {custom_mae_indian:>11.2f}y",
]

if not (np.isnan(if_mae_all) or np.isnan(custom_mae_all)):
    overall_delta = if_mae_all - custom_mae_all
    indian_delta  = if_mae_indian - custom_mae_indian
    pct_overall   = overall_delta / if_mae_all * 100
    pct_indian    = indian_delta  / if_mae_indian * 100
    lines += [
        "",
        f"Improvement (overall): {overall_delta:+.2f}y  ({pct_overall:+.1f}%)",
        f"Improvement (Indian):  {indian_delta:+.2f}y  ({pct_indian:+.1f}%)",
        "",
        "── SoP-ready sentence ──────────────────────────────────────",
        (f"I retrained the age-estimation backbone on UTKFace, oversampling "
         f"Indian faces (17% of the original dataset). On a held-out validation "
         f"split, the custom model reduced overall MAE from {if_mae_all:.1f} years "
         f"(InsightFace default) to {custom_mae_all:.1f} years, and Indian-face MAE "
         f"from {if_mae_indian:.1f} to {custom_mae_indian:.1f} years."),
    ]

result_text = "\n".join(lines)
print(result_text)

out_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(result_text)
print(f"\nSaved → {out_path}")

# ── Optional plot ─────────────────────────────────────────────────────────────
if custom_preds and insightface_preds:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, preds, trues, label, mae in [
            (axes[0], insightface_preds, insightface_trues,
             f"InsightFace default\nMAE={if_mae_all:.2f}y", if_mae_all),
            (axes[1], custom_preds,      custom_trues,
             f"Custom retrained\nMAE={custom_mae_all:.2f}y", custom_mae_all),
        ]:
            ax.scatter(trues, preds, alpha=0.3, s=6, color="#4f46e5")
            ax.plot([0, 90], [0, 90], "r--", lw=1.5, label="Perfect")
            ax.fill_between([0, 90], [-5, 85], [5, 95],
                            alpha=0.1, color="green", label="±5y band")
            ax.set_xlabel("Actual age"); ax.set_ylabel("Predicted age")
            ax.set_title(label); ax.legend(); ax.grid(alpha=0.3)

        plt.suptitle("Age Prediction: InsightFace vs Custom Model", fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "evaluation_plot.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot")
