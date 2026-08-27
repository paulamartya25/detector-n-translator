# ================================================================
# GOOGLE COLAB NOTEBOOK — Indian Face Age Estimator
# Copy each CELL into a separate Colab code cell
# Runtime → Change runtime type → T4 GPU → Save  (do this FIRST)
# ================================================================

# ════════════════════════════════════════════════════════════════
# CELL 1 — Get Kaggle API Key & Download Dataset
# ════════════════════════════════════════════════════════════════
# HOW TO GET YOUR KAGGLE API KEY:
#   1. Go to kaggle.com → Profile picture → Settings
#   2. Scroll to "API" section → Click "Create New Token"
#   3. It downloads kaggle.json — open it, copy username & key

import os

# Paste your Kaggle credentials here:
os.environ["KAGGLE_USERNAME"] = "YOUR_KAGGLE_USERNAME"   # ← change this
os.environ["KAGGLE_KEY"]      = "YOUR_KAGGLE_API_KEY"    # ← change this

# Install kaggle and download UTKFace dataset (~100MB)
get_ipython().system("pip install kaggle -q")
get_ipython().system("mkdir -p ~/.kaggle")
get_ipython().system("echo '{\"username\":\"''\",\"key\":\"''\"}' > ~/.kaggle/kaggle.json")
get_ipython().system("chmod 600 ~/.kaggle/kaggle.json")
get_ipython().system("kaggle datasets download -d jangedoo/utkface-new -p /content --unzip -q")
get_ipython().system("echo 'Dataset downloaded!'")
get_ipython().system("ls /content/UTKFace | head -5")

DATASET_PATH = "/content/UTKFace"

# ════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ════════════════════════════════════════════════════════════════
import glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU! Go to Runtime → Change runtime type → T4 GPU")

# ════════════════════════════════════════════════════════════════
# CELL 3 — Parse UTKFace Dataset
# ════════════════════════════════════════════════════════════════
# UTKFace filename: AGE_GENDER_RACE_TIMESTAMP.jpg
# RACE: 0=White, 1=Black, 2=Asian, 3=INDIAN, 4=Others

records = []
for path in glob.glob(os.path.join(DATASET_PATH, "**/*.jpg"), recursive=True):
    parts = os.path.basename(path).split("_")
    if len(parts) < 4:
        continue
    try:
        age    = int(parts[0])
        gender = int(parts[1])
        race   = int(parts[2])
        if 1 <= age <= 90:
            records.append({"path": path, "age": age,
                            "gender": gender, "race": race,
                            "is_indian": (race == 3)})
    except (ValueError, IndexError):
        continue

df = pd.DataFrame(records)
print(f"Total images : {len(df):,}")
print(f"Indian faces : {df.is_indian.sum():,}  ({df.is_indian.mean()*100:.1f}%)")
print(f"\nIndian age stats:")
print(df[df.is_indian]["age"].describe().round(1))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["age"].hist(bins=30, ax=axes[0], color="#4f46e5", alpha=0.8)
axes[0].set_title("All races — age distribution"); axes[0].set_xlabel("Age")
df[df.is_indian]["age"].hist(bins=20, ax=axes[1], color="#16a34a", alpha=0.8)
axes[1].set_title("Indian faces only"); axes[1].set_xlabel("Age")
plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════
# CELL 4 — Build Training Data (Oversample Indian 3x)
# ════════════════════════════════════════════════════════════════
indian = df[df.is_indian]
others = df[~df.is_indian]
# Indian faces appear 3x more so model learns Indian skin tones well
df_all = pd.concat([others, indian, indian, indian]).sample(frac=1, random_state=42)
train_df, val_df = train_test_split(df_all, test_size=0.15, random_state=42)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
print(f"Indian in train: {train_df.is_indian.sum():,} ({train_df.is_indian.mean()*100:.1f}%)")

# ════════════════════════════════════════════════════════════════
# CELL 5 — Dataset & DataLoader
# ════════════════════════════════════════════════════════════════
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training: heavy augmentation to handle Indian home lighting
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.5,   # simulate bright/dark Indian home lighting
        contrast=0.4,
        saturation=0.3,
        hue=0.1,
    ),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class FaceAgeDS(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.tf = TRAIN_TF if train else VAL_TF
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        return self.tf(img), torch.tensor(r.age, dtype=torch.float32)

train_ds = FaceAgeDS(train_df, train=True)
val_ds   = FaceAgeDS(val_df,   train=False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ════════════════════════════════════════════════════════════════
# CELL 6 — Model: MobileNetV2 + Age Head
# ════════════════════════════════════════════════════════════════
class AgeEstimator(nn.Module):
    """
    MobileNetV2 (ImageNet pretrained) + regression head for age.
    ~3.4M params — fast enough to run on CPU in real-time.
    """
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x).squeeze(1)

model = AgeEstimator().to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f"Model: MobileNetV2 | Parameters: {total:,}")

# ════════════════════════════════════════════════════════════════
# CELL 7 — Train! (~20-25 min on T4 GPU)
# ════════════════════════════════════════════════════════════════
criterion = nn.L1Loss()  # MAE loss — robust for age regression
EPOCHS_P1 = 5   # freeze backbone, train head
EPOCHS_P2 = 15  # unfreeze all, fine-tune
TOTAL     = EPOCHS_P1 + EPOCHS_P2

best_mae  = float("inf")
best_path = "/content/age_best.pt"
history   = {"tr": [], "vl": []}

def run(loader, train):
    model.train() if train else model.eval()
    all_p, all_t = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, ages in loader:
            imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
            p = model(imgs)
            if train:
                opt.zero_grad()
                criterion(p, ages).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            all_p += p.detach().cpu().tolist()
            all_t += ages.cpu().tolist()
    return mean_absolute_error(all_t, all_p)

# Phase 1: backbone frozen, head trains fast
for p in model.features.parameters(): p.requires_grad = False
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=3e-3, weight_decay=1e-4)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P1)

print("="*55)
print("PHASE 1 — Training head only (backbone frozen)")
print("="*55)

for ep in range(1, TOTAL + 1):
    if ep == EPOCHS_P1 + 1:
        print("\n" + "="*55)
        print("PHASE 2 — Full fine-tune (backbone unfrozen)")
        print("="*55)
        for p in model.features.parameters(): p.requires_grad = True
        opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P2, eta_min=1e-6)

    tr = run(train_loader, True)
    vl = run(val_loader,   False)
    sch.step()
    history["tr"].append(tr); history["vl"].append(vl)

    tag = ""
    if vl < best_mae:
        best_mae = vl
        torch.save(model.state_dict(), best_path)
        tag = "  ← BEST SAVED"

    print(f"Epoch {ep:2d}/{TOTAL}  Train MAE: {tr:.2f}y  Val MAE: {vl:.2f}y{tag}")

print(f"\nBest Validation MAE: {best_mae:.2f} years")

# ════════════════════════════════════════════════════════════════
# CELL 8 — Plot Training Curves
# ════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 4))
plt.plot(history["tr"], label="Train MAE", color="#4f46e5")
plt.plot(history["vl"], label="Val MAE",   color="#16a34a")
plt.axvline(EPOCHS_P1 - 1, color="red", linestyle="--",
            alpha=0.6, label="Phase 2 starts")
plt.xlabel("Epoch"); plt.ylabel("MAE (years)")
plt.title(f"Training History — Best Val MAE: {best_mae:.2f} years")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("/content/training_history.png", dpi=100)
plt.show()

# ════════════════════════════════════════════════════════════════
# CELL 9 — Indian Face Evaluation
# ════════════════════════════════════════════════════════════════
model.load_state_dict(torch.load(best_path))
model.eval()

test_indian = df[df.is_indian].sample(min(300, len(df[df.is_indian])), random_state=99)
actual, pred = [], []

with torch.no_grad():
    for _, r in test_indian.iterrows():
        img = VAL_TF(Image.open(r.path).convert("RGB")).unsqueeze(0).to(DEVICE)
        pred.append(float(model(img)))
        actual.append(r.age)

indian_mae  = mean_absolute_error(actual, pred)
bias        = np.mean(pred) - np.mean(actual)
print(f"\n{'='*45}")
print(f"Indian-face evaluation ({len(actual)} images):")
print(f"  MAE  : {indian_mae:.2f} years")
print(f"  Bias : {bias:+.1f}y  ({'overestimates' if bias>0 else 'underestimates'})")
print(f"  Mean actual:    {np.mean(actual):.1f}y")
print(f"  Mean predicted: {np.mean(pred):.1f}y")
print(f"{'='*45}")

plt.figure(figsize=(7, 6))
plt.scatter(actual, pred, alpha=0.4, color="#16a34a", s=20)
plt.plot([0, 90], [0, 90], "r--", lw=1.5, label="Perfect")
plt.xlabel("Actual Age"); plt.ylabel("Predicted Age")
plt.title(f"Indian Faces: MAE={indian_mae:.1f}y | Bias={bias:+.1f}y")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("/content/indian_eval.png", dpi=100)
plt.show()

# ════════════════════════════════════════════════════════════════
# CELL 10 — Export to ONNX & Download
# ════════════════════════════════════════════════════════════════
get_ipython().system("pip install onnx onnxruntime -q")
import onnxruntime as ort

model.load_state_dict(torch.load(best_path))
model.eval()

dummy     = torch.randn(1, 3, 224, 224).to(DEVICE)
onnx_path = "/content/age_estimator_indian.onnx"

torch.onnx.export(
    model, dummy, onnx_path,
    opset_version=12,
    input_names=["face_crop"],
    output_names=["age"],
    dynamic_axes={"face_crop": {0: "batch"}, "age": {0: "batch"}},
)

# Verify ONNX
sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_in = np.random.randn(1, 3, 224, 224).astype(np.float32)
out     = sess.run(["age"], {"face_crop": test_in})[0]
size_mb = os.path.getsize(onnx_path) / 1e6

print(f"\nONNX export verified!")
print(f"  Output: {out[0]:.1f}y (random input)")
print(f"  Size  : {size_mb:.1f} MB")

# Download the model to your PC
from google.colab import files
files.download(onnx_path)
print("\nDownloading age_estimator_indian.onnx to your PC...")
print("NEXT STEP: Copy it to your project's  models/  folder!")
