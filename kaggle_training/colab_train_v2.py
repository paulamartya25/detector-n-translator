# ================================================================
# DETECTOR N TRANSLATOR — Age Model v2 Training (EfficientNetB3)
# Target: 85-90% age group accuracy | MAE ~3-4 years
#
# HOW TO USE:
#   1. Open Google Colab: colab.research.google.com
#   2. Runtime -> Change runtime type -> T4 GPU -> Save
#   3. Paste this ENTIRE file into one cell and run it
#   4. Replace YOUR_KAGGLE_KEY_HERE with your key from kaggle.json
#   5. Wait ~35 min -> age_best_v2.pt auto-downloads to your PC
#   6. Copy downloaded file to:
#        models/age_best_v2.pt   (rename to age_best.pt to activate)
# ================================================================

import os, glob, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

# ================================================================
# STEP 1 — CONFIGURATION (edit these if needed)
# ================================================================

KAGGLE_USERNAME = "amartyapaul7"
KAGGLE_KEY      = "YOUR_KAGGLE_KEY_HERE"    # <-- paste your key here

DATASET_PATH    = "/content/UTKFace"
BEST_PT_PATH    = "/content/age_best_v2.pt"

EPOCHS_PHASE1   = 5    # backbone frozen  (fast head training)
EPOCHS_PHASE2   = 35   # full fine-tune   (deep convergence)
TOTAL_EPOCHS    = EPOCHS_PHASE1 + EPOCHS_PHASE2   # = 40

BATCH_SIZE      = 32   # lower than v1 (EfficientNetB3 is bigger)
INPUT_SIZE      = 260  # larger input = more facial detail
LR_PHASE1       = 1e-3
LR_PHASE2       = 3e-5
WEIGHT_DECAY    = 1e-4
INDIAN_OVERSAMPLE = 3  # Indian faces repeated Nx
TEEN_OVERSAMPLE   = 5  # Teen faces repeated Nx (was weakest class)
MIDDLE_OVERSAMPLE = 3  # MiddleAge faces repeated Nx

# ================================================================
# STEP 2 — INSTALL AND DOWNLOAD DATASET
# ================================================================

os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

os.system("pip install kaggle torchvision scikit-learn matplotlib -q")
os.system("mkdir -p ~/.kaggle")
os.system(f'echo \'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}\' > ~/.kaggle/kaggle.json')
os.system("chmod 600 ~/.kaggle/kaggle.json")

if not os.path.exists(DATASET_PATH):
    print("Downloading UTKFace dataset...")
    os.system("kaggle datasets download -d jangedoo/utkface-new -p /content --unzip -q")
    print("Dataset downloaded!")
else:
    print("Dataset already present, skipping download.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
if DEVICE != "cuda":
    print("WARNING: No GPU detected!")
    print("Go to: Runtime -> Change runtime type -> T4 GPU -> Save")
    print("Then re-run this cell.")

# ================================================================
# STEP 3 — AGE GROUP DEFINITIONS
# ================================================================
# These 5 groups map the regression output to classification labels
# The model learns BOTH the exact age AND which group it belongs to
# This dual training forces better boundary discrimination

def age_to_group(age):
    if   age <= 12: return 0   # Child
    elif age <= 19: return 1   # Teen        <- was weakest (F1=0.349 in v1)
    elif age <= 35: return 2   # YoungAdult
    elif age <= 55: return 3   # MiddleAge   <- was weak (F1=0.650 in v1)
    else:           return 4   # Senior

GROUP_NAMES = [
    "Child (0-12)",
    "Teen (13-19)",
    "YoungAdult (20-35)",
    "MiddleAge (36-55)",
    "Senior (56+)",
]
N_GROUPS = len(GROUP_NAMES)

# ================================================================
# STEP 4 — PARSE UTKFACE DATASET
# ================================================================
# UTKFace filename format: AGE_GENDER_RACE_timestamp.jpg
# RACE codes: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

print("\nParsing UTKFace dataset...")
records = []
for path in glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True):
    parts = os.path.basename(path).split("_")
    if len(parts) < 4:
        continue
    try:
        age  = int(parts[0])
        race = int(parts[2])
        if 1 <= age <= 90:
            group = age_to_group(age)
            records.append({
                "path":      path,
                "age":       age,
                "race":      race,
                "group":     group,
                "is_indian": (race == 3),
                "is_teen":   (group == 1),
                "is_middle": (group == 3),
            })
    except:
        continue

df = pd.DataFrame(records)
print(f"Total images: {len(df):,}")
print(f"Indian faces: {df.is_indian.sum():,} ({df.is_indian.mean()*100:.1f}%)")
print("\nGroup distribution (before oversampling):")
for g, name in enumerate(GROUP_NAMES):
    n = (df.group == g).sum()
    bar = "#" * (n // 100)
    print(f"  {name:20s}: {n:5,}  {bar}")

# ================================================================
# STEP 5 — SMART OVERSAMPLING
# ================================================================
# Strategy:
#   - Indian faces 3x  (domain adaptation for Indian faces)
#   - Teen faces 5x    (fixes terrible F1=0.349 from v1)
#   - MiddleAge faces 3x (fixes weak F1=0.650 from v1)
# Note: Some Indian teens/middle-aged get boosted by both multipliers

indian  = df[df.is_indian]
others  = df[~df.is_indian]
teens   = df[df.is_teen]
middles = df[df.is_middle]

parts = [others]
# Indian oversampling
for _ in range(INDIAN_OVERSAMPLE):
    parts.append(indian)
# Teen oversampling (extra on top of indian overlap)
for _ in range(TEEN_OVERSAMPLE):
    parts.append(teens)
# MiddleAge oversampling
for _ in range(MIDDLE_OVERSAMPLE):
    parts.append(middles)

df_all = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
train_df, val_df = train_test_split(df_all, test_size=0.15, random_state=42)

print(f"\nAfter oversampling:")
print(f"  Total: {len(df_all):,}")
print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

# ================================================================
# STEP 6 — DATA AUGMENTATION
# ================================================================
# Training: heavy augmentation (brightness, contrast, rotation, perspective)
# This simulates: different lighting, cameras, angles, Indian skin tones
# Validation: no augmentation (clean test)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((int(INPUT_SIZE * 1.15), int(INPUT_SIZE * 1.15))),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.5,   # simulate different lighting
        contrast=0.5,
        saturation=0.4,
        hue=0.1,
    ),
    transforms.RandomRotation(degrees=20),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # hide small patches
])

VAL_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ================================================================
# STEP 7 — PYTORCH DATASET
# ================================================================

class FaceAgeDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.tf = TRAIN_TF if train else VAL_TF

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        return (
            self.tf(img),
            torch.tensor(r.age,   dtype=torch.float32),  # exact age (regression target)
            torch.tensor(r.group, dtype=torch.long),      # age group (classification target)
        )

train_loader = DataLoader(
    FaceAgeDS(train_df, True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True,
)
val_loader = DataLoader(
    FaceAgeDS(val_df, False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True,
)
print(f"\nDataLoaders ready")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")

# ================================================================
# STEP 8 — MODEL ARCHITECTURE: EfficientNetB3 + Dual Head
# ================================================================
# Why EfficientNetB3 over MobileNetV2 (v1)?
#   - Better accuracy: ImageNet top-1 = 81.6% vs 71.8%
#   - Better feature extraction for age estimation
#   - Still CPU-compatible (14MB weights vs 13MB for MobileNetV2)
#
# Dual Head design:
#   - Regression head -> predicts exact age in years (L1 loss)
#   - Classification head -> predicts age group (CrossEntropy loss)
#   - Combined loss = L1 + 0.5 * CrossEntropy
#   - Classification head forces model to learn sharp group boundaries
#   - This specifically fixes the Teen/YoungAdult confusion from v1

class DualHeadAgeNet(nn.Module):
    def __init__(self, n_groups=N_GROUPS):
        super().__init__()
        base = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        # EfficientNetB3 feature extractor (outputs 1536-dim vector)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Regression head: predicts exact age (years)
        self.reg_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Classification head: predicts age group
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 256),
            nn.GELU(),
            nn.Linear(256, n_groups),
        )

    def forward(self, x):
        feat = self.pool(self.features(x)).flatten(1)   # (B, 1536)
        age  = self.reg_head(feat).squeeze(1)            # (B,) exact age
        grp  = self.cls_head(feat)                       # (B, N_GROUPS) logits
        return age, grp

model = DualHeadAgeNet().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: EfficientNetB3 + Dual Head")
print(f"Parameters: {total_params:,}")
print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")

# ================================================================
# STEP 9 — LOSS FUNCTIONS
# ================================================================
# L1 loss for regression (age in years, robust to outliers)
# Weighted CrossEntropy for classification:
#   Teen gets 4x weight  (was terrible in v1)
#   Senior gets 2x weight (slightly weak)
#   Others get 1x weight

reg_criterion = nn.L1Loss()

class_weights = torch.tensor(
    [1.0, 4.0, 1.0, 3.0, 2.0],  # Child, Teen, YoungAdult, MiddleAge, Senior
    dtype=torch.float32,
).to(DEVICE)
cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

# ================================================================
# STEP 10 — TRAINING LOOP
# ================================================================

best_acc    = 0.0
best_mae_v  = float("inf")
history     = {"tr_mae": [], "vl_mae": [], "vl_acc": []}

def run_epoch(loader, training):
    model.train() if training else model.eval()
    all_age_pred, all_age_true = [], []
    all_grp_pred, all_grp_true = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, ages, groups in loader:
            imgs, ages, groups = imgs.to(DEVICE), ages.to(DEVICE), groups.to(DEVICE)

            age_pred, grp_pred = model(imgs)

            # Combined loss: regression + 0.5 * classification
            loss_reg = reg_criterion(age_pred, ages)
            loss_cls = cls_criterion(grp_pred, groups)
            loss     = loss_reg + 0.5 * loss_cls

            if training:
                opt.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                opt.step()

            all_age_pred += age_pred.detach().cpu().tolist()
            all_age_true += ages.cpu().tolist()
            all_grp_pred += grp_pred.argmax(1).detach().cpu().tolist()
            all_grp_true += groups.cpu().tolist()

    mae = mean_absolute_error(all_age_true, all_age_pred)
    acc = accuracy_score(all_grp_true, all_grp_pred)
    return mae, acc

# ================================================================
# TRAINING — Phase 1: backbone frozen (fast head convergence)
# ================================================================
for p in model.features.parameters():
    p.requires_grad = False

trainable = filter(lambda p: p.requires_grad, model.parameters())
opt = optim.AdamW(trainable, lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=EPOCHS_PHASE1, T_mult=1)

print("\n" + "="*60)
print("  TRAINING STARTED -- ~35 min on T4 GPU")
print("="*60)

for ep in range(1, TOTAL_EPOCHS + 1):

    # Switch to Phase 2 at the right epoch
    if ep == EPOCHS_PHASE1 + 1:
        print("\n--- Phase 2: Full fine-tune (all layers) ---\n")
        for p in model.features.parameters():
            p.requires_grad = True
        opt = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=1e-7
        )

    tr_mae, tr_acc = run_epoch(train_loader, True)
    vl_mae, vl_acc = run_epoch(val_loader,   False)
    sch.step()

    history["tr_mae"].append(tr_mae)
    history["vl_mae"].append(vl_mae)
    history["vl_acc"].append(vl_acc)

    # Save best model (based on val accuracy, not MAE)
    tag = ""
    if vl_acc > best_acc:
        best_acc   = vl_acc
        best_mae_v = vl_mae
        torch.save(model.state_dict(), BEST_PT_PATH)
        tag = "  <-- BEST SAVED"

    print(
        f"Ep {ep:2d}/{TOTAL_EPOCHS}  |  "
        f"Train: MAE={tr_mae:.1f}y  Acc={tr_acc*100:.1f}%  |  "
        f"Val:   MAE={vl_mae:.1f}y  Acc={vl_acc*100:.1f}%"
        f"{tag}"
    )

print(f"\nBest Val: Accuracy={best_acc*100:.1f}%  MAE={best_mae_v:.2f}y")

# ================================================================
# STEP 11 — EVALUATION PLOTS
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# MAE curve
axes[0].plot(history["tr_mae"], label="Train MAE", color="#4f46e5", lw=2)
axes[0].plot(history["vl_mae"], label="Val MAE",   color="#16a34a", lw=2)
axes[0].axvline(EPOCHS_PHASE1 - 1, color="red", linestyle="--", alpha=0.5, label="Phase 2 start")
axes[0].set_title("Age MAE (years)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE (years)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Accuracy curve
axes[1].plot([a * 100 for a in history["vl_acc"]], color="#16a34a", lw=2, label="Val Accuracy")
axes[1].axhline(85, color="orange", linestyle="--", lw=1.5, label="85% target")
axes[1].axhline(90, color="red",    linestyle="--", lw=1.5, label="90% target")
axes[1].set_title("Age Group Accuracy (%)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy %")
axes[1].legend(); axes[1].grid(alpha=0.3)

# Actual vs predicted scatter (on val set)
model.load_state_dict(torch.load(BEST_PT_PATH))
model.eval()
actual_ages, pred_ages = [], []
with torch.no_grad():
    for imgs, ages, _ in val_loader:
        age_pred, _ = model(imgs.to(DEVICE))
        actual_ages += ages.tolist()
        pred_ages   += age_pred.cpu().tolist()

actual_ages = np.array(actual_ages)
pred_ages   = np.clip(np.array(pred_ages), 1, 90)
mae_final   = mean_absolute_error(actual_ages, pred_ages)
bias_final  = np.mean(pred_ages - actual_ages)

axes[2].scatter(actual_ages, pred_ages, alpha=0.3, s=8, color="#4f46e5")
axes[2].plot([0, 90], [0, 90], "r--", lw=1.5, label="Perfect")
axes[2].fill_between([0, 90], [0-5, 90-5], [0+5, 90+5],
                      alpha=0.1, color="green", label="+-5y band")
axes[2].set_title(f"Actual vs Predicted  (MAE={mae_final:.1f}y, Bias={bias_final:+.1f}y)")
axes[2].set_xlabel("Actual Age"); axes[2].set_ylabel("Predicted Age")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f"EfficientNetB3 Age Model v2 -- Best Acc: {best_acc*100:.1f}%  MAE: {best_mae_v:.2f}y",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/training_results_v2.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /content/training_results_v2.png")

# ================================================================
# STEP 12 — FULL CLASSIFICATION REPORT
# ================================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def age_to_group_arr(ages):
    return np.array([age_to_group(int(a)) for a in ages])

actual_grps = age_to_group_arr(actual_ages)
pred_grps   = age_to_group_arr(pred_ages)

print("\n" + "="*60)
print("  CLASSIFICATION REPORT")
print("="*60)
print(classification_report(actual_grps, pred_grps,
                             target_names=GROUP_NAMES, digits=3))

cm = confusion_matrix(actual_grps, pred_grps)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=GROUP_NAMES, yticklabels=GROUP_NAMES,
            linewidths=0.5)
plt.title(f"Confusion Matrix -- Age Group Classification v2\nAccuracy: {best_acc*100:.1f}%")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("/content/confusion_matrix_v2.png", dpi=120)
plt.show()

# ================================================================
# STEP 13 — DOWNLOAD MODEL + PLOTS TO YOUR PC
# ================================================================

from google.colab import files

print("\nDownloading files to your PC...")
files.download(BEST_PT_PATH)
files.download("/content/training_results_v2.png")
files.download("/content/confusion_matrix_v2.png")

print("\n" + "="*60)
print("  DONE!")
print("="*60)
print(f"  Best Accuracy: {best_acc*100:.1f}%")
print(f"  Best MAE:      {best_mae_v:.2f} years")
print()
print("  Next steps:")
print("  1. Move downloaded age_best_v2.pt to your project:")
print("       models\\age_best_v2.pt")
print("  2. Rename it to age_best.pt (replaces old model):")
print("       models\\age_best.pt")
print("  3. Restart the app:")
print("       python app.py")
print("  4. Terminal should show:")
print("       [FaceAnalyzer] Custom age model (PyTorch .pt) loaded")