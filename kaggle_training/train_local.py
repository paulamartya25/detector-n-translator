"""
train_local.py  --  Age Model v2  --  RTX 3050 Optimised
==========================================================
Run from the project root folder:
    python kaggle_training/train_local.py

Key optimisations vs colab_train_v2.py:
  AMP  (float16 mixed precision)  -> halves VRAM: 4GB -> ~2GB peak
  Gradient accumulation (steps=2) -> simulates batch 32 with batch 16
  Auto-checkpoint every epoch     -> safe to interrupt (Ctrl+C), resume next time
  Live ETA in terminal            -> no browser needed
  Writes best model straight to   -> models/age_best_v2.pt

After training:
    copy models\\age_best_v2.pt models\\age_best.pt
    python app.py
==========================================================
"""

import os, sys, glob, time, json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

import matplotlib
matplotlib.use("Agg")           # no GUI needed for plots
import matplotlib.pyplot as plt

# ================================================================
# CONFIGURATION  --  edit KAGGLE_KEY before first run
# ================================================================

KAGGLE_USERNAME   = "amartyapaul7"
KAGGLE_KEY        = "YOUR_KAGGLE_KEY_HERE"   # paste key from kaggle.json

DATASET_PATH      = "data/UTKFace"
CHECKPOINT_PATH   = "kaggle_training/checkpoints/train_local_ckpt.pt"
BEST_MODEL_PATH   = "models/age_best_v2.pt"
PLOT_PATH         = "outputs/training_results_v2.png"

INPUT_SIZE        = 260     # EfficientNetB3 native resolution
BATCH_SIZE        = 16      # safe for RTX 3050 4GB VRAM with AMP
GRAD_ACCUM        = 2       # effective batch = 16 * 2 = 32
EPOCHS_PHASE1     = 5       # backbone frozen, train heads only
EPOCHS_PHASE2     = 35      # full fine-tune
TOTAL_EPOCHS      = EPOCHS_PHASE1 + EPOCHS_PHASE2   # 40 total
LR_PHASE1         = 1e-3
LR_PHASE2         = 3e-5
WEIGHT_DECAY      = 1e-4
NUM_WORKERS       = 4       # Windows: max stable workers

INDIAN_OS         = 3       # Indian oversampling multiplier
TEEN_OS           = 5       # Teen oversampling  (worst class in v1: F1=0.349)
MIDDLE_OS         = 3       # MiddleAge oversampling

# ================================================================
# SETUP
# ================================================================

os.makedirs("data",                                         exist_ok=True)
os.makedirs("models",                                       exist_ok=True)
os.makedirs("outputs",                                      exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH),               exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

print("=" * 65)
print("  Detector N Translator -- Age Model v2 -- Local Training")
print("=" * 65)
print(f"  Device  : {DEVICE}")
if DEVICE.type == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1024**3
    print(f"  GPU     : {gpu.name}  ({vram_gb:.1f} GB VRAM)")
    print(f"  CUDA    : {torch.version.cuda}")
else:
    print("  WARNING : No GPU found. Training will be very slow (~8h).")
    print("            Use Colab (colab_train_v2.py) if this is the case.")
print(f"  PyTorch : {torch.__version__}")
print(f"  AMP     : {USE_AMP}  (float16 mixed precision)")
print(f"  Eff. batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
print("=" * 65)

# ================================================================
# STEP 1 -- DATASET DOWNLOAD
# ================================================================

img_count = len(glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True))

if img_count < 1000:
    if KAGGLE_KEY == "YOUR_KAGGLE_KEY_HERE":
        print()
        print("ERROR: Set your Kaggle API key first!")
        print("  1. Go to kaggle.com -> Your Profile -> Settings -> API")
        print("  2. Click 'Create New Token' -> downloads kaggle.json")
        print("  3. Open kaggle.json and copy the 'key' value")
        print("  4. Paste it into KAGGLE_KEY in this file (line ~40)")
        sys.exit(1)

    print(f"\nDownloading UTKFace dataset (~160 MB)...")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    key_content = json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY})
    key_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(key_path, "w") as f:
        f.write(key_content)
    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass  # Windows doesn't support chmod, that's fine

    ret = os.system("pip install kaggle -q")
    ret = os.system(f"kaggle datasets download -d jangedoo/utkface-new -p data --unzip -q")
    if ret != 0:
        print("Download failed. Check your KAGGLE_KEY and internet connection.")
        sys.exit(1)
    img_count = len(glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True))
    print(f"Done! {img_count:,} images downloaded.\n")
else:
    print(f"\nDataset found: {img_count:,} images. Skipping download.")

# ================================================================
# STEP 2 -- AGE GROUP DEFINITIONS  (matches face_analyzer.py)
# ================================================================

GROUP_NAMES = [
    "Child(0-12)",
    "Teen(13-19)",
    "YoungAdult(20-35)",
    "MiddleAge(36-55)",
    "Senior(56+)",
]
N_GROUPS = len(GROUP_NAMES)


def age_to_group(age: int) -> int:
    if   age <= 12: return 0
    elif age <= 19: return 1
    elif age <= 35: return 2
    elif age <= 55: return 3
    else:           return 4

# ================================================================
# STEP 3 -- PARSE DATASET
# ================================================================

print("\nParsing UTKFace dataset (filename format: AGE_GENDER_RACE_*.jpg) ...")
records = []
for path in glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True):
    parts = os.path.basename(path).split("_")
    if len(parts) < 4:
        continue
    try:
        age  = int(parts[0])
        race = int(parts[2])
        if not (1 <= age <= 90):
            continue
        g = age_to_group(age)
        records.append({
            "path":      path,
            "age":       age,
            "race":      race,
            "group":     g,
            "is_indian": (race == 3),
            "is_teen":   (g == 1),
            "is_middle": (g == 3),
        })
    except (ValueError, IndexError):
        continue

df = pd.DataFrame(records)
print(f"  Valid images  : {len(df):,}")
print(f"  Indian faces  : {df.is_indian.sum():,}  ({df.is_indian.mean()*100:.1f}%)")
print("\n  Group distribution (before oversampling):")
for g, name in enumerate(GROUP_NAMES):
    n   = int((df.group == g).sum())
    bar = "#" * (n // 150)
    print(f"    {name:22s}: {n:5,}  {bar}")

# ================================================================
# STEP 4 -- SMART OVERSAMPLING
# ================================================================

indian  = df[df.is_indian]
others  = df[~df.is_indian]
teens   = df[df.is_teen]
middles = df[df.is_middle]

parts = [others]
parts += [indian]  * INDIAN_OS
parts += [teens]   * TEEN_OS
parts += [middles] * MIDDLE_OS

df_all                  = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
train_df, val_df        = train_test_split(df_all, test_size=0.15, random_state=42)
print(f"\n  After oversampling -> Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# ================================================================
# STEP 5 -- AUGMENTATION PIPELINES
# ================================================================

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((int(INPUT_SIZE * 1.15), int(INPUT_SIZE * 1.15))),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=20),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

VAL_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ================================================================
# STEP 6 -- DATASET CLASS
# ================================================================

class FaceAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.tf = TRAIN_TF if train else VAL_TF

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            img = Image.open(row.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (128, 128, 128))
        img_t   = self.tf(img)
        age_t   = torch.tensor(row.age,   dtype=torch.float32)
        group_t = torch.tensor(row.group, dtype=torch.long)
        return img_t, age_t, group_t


train_ds = FaceAgeDataset(train_df, train=True)
val_ds   = FaceAgeDataset(val_df,   train=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP, persistent_workers=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP, persistent_workers=True,
)
print(f"  Train batches : {len(train_loader)}")
print(f"  Val   batches : {len(val_loader)}")

# ================================================================
# STEP 7 -- MODEL  (must exactly match face_analyzer.py's _DualHeadAgeNet)
# ================================================================

class DualHeadAgeNet(nn.Module):
    """
    EfficientNetB3 backbone + two heads:
      reg_head  -> scalar age in years   (regression, L1 loss)
      cls_head  -> 5-class age group     (classification, weighted CE loss)

    Architecture is mirrored exactly in face_analyzer.py _try_load_custom_age_model()
    so the saved weights load without any key mismatches.
    """
    def __init__(self):
        super().__init__()
        base          = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.features = base.features          # shared backbone (1536-dim output)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.GELU(),
            nn.Linear(128, 1),
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 256), nn.GELU(),
            nn.Linear(256, N_GROUPS),
        )

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)   # (B, 1536)
        age  = self.reg_head(feat).squeeze(1)            # (B,)  float
        grp  = self.cls_head(feat)                       # (B, 5) logits
        return age, grp


model   = DualHeadAgeNet().to(DEVICE)
n_param = sum(p.numel() for p in model.parameters())
print(f"\n  Model: EfficientNetB3 + Dual Head  ({n_param:,} parameters)")

# ================================================================
# STEP 8 -- LOSS FUNCTIONS + AMP SCALER
# ================================================================

reg_criterion = nn.L1Loss()

# Weighted CE: Teen=4x, MiddleAge=3x, Senior=2x
# (these were the three worst classes in v1)
cls_weights   = torch.tensor([1.0, 4.0, 1.0, 3.0, 2.0], dtype=torch.float32).to(DEVICE)
cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)

scaler        = GradScaler("cuda" if USE_AMP else "cpu", enabled=USE_AMP)

# ================================================================
# STEP 9 -- CHECKPOINT HELPERS
# ================================================================

def _save_ckpt(epoch: int, model, opt, sch, best_acc: float, history: dict):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "opt":       opt.state_dict(),
        "sch":       sch.state_dict(),
        "best_acc":  best_acc,
        "history":   history,
    }, CHECKPOINT_PATH)


def _load_ckpt(model, opt, sch):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, 0.0, {"tr_mae": [], "vl_mae": [], "vl_acc": []}
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sch.load_state_dict(ckpt["sch"])
    ep = ckpt["epoch"]
    print(f"\n  Resuming from checkpoint at epoch {ep + 1}")
    print(f"  Best accuracy so far: {ckpt['best_acc']*100:.1f}%")
    return ep + 1, ckpt["best_acc"], ckpt["history"]

# ================================================================
# STEP 10 -- EPOCH RUNNER
# ================================================================

def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def run_epoch(loader, training: bool, opt=None) -> tuple:
    model.train() if training else model.eval()
    age_pred_all, age_true_all = [], []
    grp_pred_all, grp_true_all = [], []

    if training:
        opt.zero_grad()

    for step, (imgs, ages, groups) in enumerate(loader):
        imgs   = imgs.to(DEVICE)
        ages   = ages.to(DEVICE)
        groups = groups.to(DEVICE)

        if training:
            with autocast("cuda" if USE_AMP else "cpu", enabled=USE_AMP):
                age_p, grp_p = model(imgs)
                loss = (reg_criterion(age_p, ages) + 0.5 * cls_criterion(grp_p, groups)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        else:
            with torch.no_grad(), autocast("cuda" if USE_AMP else "cpu", enabled=USE_AMP):
                age_p, grp_p = model(imgs)

        age_pred_all += age_p.detach().float().cpu().tolist()
        age_true_all += ages.cpu().tolist()
        grp_pred_all += grp_p.argmax(1).detach().cpu().tolist()
        grp_true_all += groups.cpu().tolist()

    mae = mean_absolute_error(age_true_all, age_pred_all)
    acc = accuracy_score(grp_true_all, grp_pred_all)
    return mae, acc

# ================================================================
# STEP 11 -- PHASE 1 SETUP  (backbone frozen)
# ================================================================

for p in model.features.parameters():
    p.requires_grad = False

opt = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1, weight_decay=WEIGHT_DECAY,
)
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=EPOCHS_PHASE1)

start_epoch, best_acc, history = _load_ckpt(model, opt, sch)

print()
print("=" * 65)
print(f"  TRAINING START  --  {TOTAL_EPOCHS} epochs total")
print(f"  Phase 1: Ep 1-{EPOCHS_PHASE1}   backbone FROZEN   LR={LR_PHASE1}")
print(f"  Phase 2: Ep {EPOCHS_PHASE1+1}-{TOTAL_EPOCHS}  full fine-tune  LR={LR_PHASE2}")
print("=" * 65)

best_mae  = float("inf")
t_start   = time.time()

# ================================================================
# STEP 12 -- MAIN TRAINING LOOP
# ================================================================

try:
    for ep in range(start_epoch, TOTAL_EPOCHS):

        # ── Phase 2 switch ──────────────────────────────────────
        if ep == EPOCHS_PHASE1:
            print()
            print("-" * 65)
            print("  PHASE 2: Unfreezing backbone for full fine-tune")
            print("-" * 65)
            for p in model.features.parameters():
                p.requires_grad = True
            opt = optim.AdamW(
                model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY
            )
            sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=10, T_mult=2, eta_min=1e-7
            )

        t_ep           = time.time()
        tr_mae, tr_acc = run_epoch(train_loader, True,  opt)
        vl_mae, vl_acc = run_epoch(val_loader,   False)
        sch.step()

        ep_secs    = time.time() - t_ep
        elapsed    = time.time() - t_start
        remaining  = ep_secs * (TOTAL_EPOCHS - ep - 1)
        lr_now     = opt.param_groups[0]["lr"]

        history["tr_mae"].append(tr_mae)
        history["vl_mae"].append(vl_mae)
        history["vl_acc"].append(vl_acc)

        tag = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_mae = vl_mae
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            tag = "  <-- BEST  saved to models/age_best_v2.pt"

        print(
            f"Ep {ep+1:2d}/{TOTAL_EPOCHS}"
            f"  Tr {tr_mae:.1f}y {tr_acc*100:.1f}%"
            f"  |  Val {vl_mae:.1f}y {vl_acc*100:.1f}%"
            f"  LR={lr_now:.1e}"
            f"  [{fmt_time(ep_secs)}]"
            f"  ETA {fmt_time(remaining)}"
            f"{tag}"
        )

        _save_ckpt(ep, model, opt, sch, best_acc, history)

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Checkpoint saved -- re-run to resume.")
    sys.exit(0)

# ================================================================
# STEP 13 -- FINAL EVALUATION
# ================================================================

print()
print("=" * 65)
print("  TRAINING COMPLETE")
print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
print(f"  Best Val MAE      : {best_mae:.2f} years")
print(f"  Model saved to    : {BEST_MODEL_PATH}")
print("=" * 65)

# Load best weights for evaluation
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()
act_ages, prd_ages, act_grps, prd_grps = [], [], [], []

with torch.no_grad(), autocast("cuda" if USE_AMP else "cpu", enabled=USE_AMP):
    for imgs, ages, groups in val_loader:
        ap, gp = model(imgs.to(DEVICE))
        act_ages += ages.tolist()
        prd_ages += ap.float().cpu().tolist()
        act_grps += groups.tolist()
        prd_grps += gp.argmax(1).cpu().tolist()

act_ages = np.array(act_ages)
prd_ages = np.clip(np.array(prd_ages), 1, 90)
bias     = float(np.mean(prd_ages - act_ages))

print(f"\n  Final MAE  : {mean_absolute_error(act_ages, prd_ages):.2f} years")
print(f"  Final Bias : {bias:+.2f} years")
print()
print(classification_report(act_grps, prd_grps, target_names=GROUP_NAMES, digits=3))

# ================================================================
# STEP 14 -- PLOTS
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["tr_mae"], label="Train MAE", color="#4f46e5", lw=2)
axes[0].plot(history["vl_mae"], label="Val MAE",   color="#16a34a", lw=2)
axes[0].axvline(EPOCHS_PHASE1 - 1, color="red", ls="--", alpha=0.5, label="Phase 2 start")
axes[0].set_title("MAE (years)"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot([a * 100 for a in history["vl_acc"]], color="#16a34a", lw=2)
axes[1].axhline(85, color="orange", ls="--", lw=1.5, label="85% target")
axes[1].axhline(90, color="red",    ls="--", lw=1.5, label="90% target")
axes[1].set_title("Val Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].scatter(act_ages, prd_ages, alpha=0.3, s=8, color="#4f46e5")
axes[2].plot([0, 90], [0, 90], "r--", lw=1.5, label="Perfect")
axes[2].fill_between([0,90],[0-5,90-5],[0+5,90+5], alpha=0.1, color="green", label="+-5y band")
axes[2].set_title(f"Actual vs Predicted (MAE={mean_absolute_error(act_ages, prd_ages):.1f}y, Bias={bias:+.1f}y)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f"EfficientNetB3 v2  --  Best Acc={best_acc*100:.1f}%  MAE={best_mae:.2f}y", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
print(f"\n  Plot saved: {PLOT_PATH}")

# Cleanup checkpoint after successful full run
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

# ================================================================
# STEP 15 -- ACTIVATION INSTRUCTIONS
# ================================================================

print()
print("=" * 65)
print("  NEXT STEPS TO ACTIVATE IN APP")
print("=" * 65)
print()
print("  1. Copy v2 model over the old one:")
print("       copy models\\age_best_v2.pt models\\age_best.pt")
print()
print("  2. Run the app:")
print("       python app.py")
print()
print("  3. Terminal should show:")
print("       [FaceAnalyzer] Detected v2 model (EfficientNetB3, ~45 MB)")
print("       [FaceAnalyzer] Custom age model (EfficientNetB3 v2, ~45 MB) loaded")
print()
