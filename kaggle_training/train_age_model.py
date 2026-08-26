# ============================================================
# KAGGLE NOTEBOOK — Indian Face Age Estimator
# Dataset : UTKFace (race=3 = Indian) + all races
# Model   : MobileNetV2 pretrained → Age Regression Head
# Output  : age_estimator_indian.onnx
# ============================================================
#
# SETUP ON KAGGLE:
#   1. kaggle.com → New Notebook → Python
#   2. Settings → Accelerator → GPU T4
#   3. Add Dataset → search "utkface-new" by jangedoo → Add
#   4. Run all cells in order
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import os, glob, numpy as np, pandas as pd
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

# ── CELL 2: Load UTKFace ─────────────────────────────────────
# UTKFace filename: AGE_GENDER_RACE_TIMESTAMP.jpg
# RACE labels: 0=White 1=Black 2=Asian 3=INDIAN 4=Others
DATASET_PATH = "/kaggle/input/utkface-new/UTKFace"

records = []
for path in glob.glob(os.path.join(DATASET_PATH, "*.jpg")):
    parts = os.path.basename(path).split("_")
    if len(parts) < 4:
        continue
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
        if 1 <= age <= 90:
            records.append({"path": path, "age": age,
                            "gender": gender, "race": race,
                            "is_indian": (race == 3)})
    except (ValueError, IndexError):
        continue

df = pd.DataFrame(records)
print(f"Total: {len(df):,} | Indian: {df.is_indian.sum():,} ({df.is_indian.mean()*100:.1f}%)")

# Age distribution
df["age"].hist(bins=30, color="#4f46e5", alpha=0.7, figsize=(8,3))
plt.title("Age distribution — all UTKFace")
plt.xlabel("Age"); plt.savefig("dist_all.png"); plt.show()
df[df.is_indian]["age"].hist(bins=20, color="#16a34a", alpha=0.7, figsize=(8,3))
plt.title("Age distribution — Indian faces only")
plt.xlabel("Age"); plt.savefig("dist_indian.png"); plt.show()

# ── CELL 3: Oversample Indian faces 3x ──────────────────────
indian = df[df.is_indian]
others = df[~df.is_indian]
df_all = pd.concat([others, indian, indian, indian]).sample(frac=1, random_state=42)
train_df, val_df = train_test_split(df_all, test_size=0.15, random_state=42)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

# ── CELL 4: Dataset ──────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomRotation(15),
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
    def __len__(self):  return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        return self.tf(img), torch.tensor(r.age, dtype=torch.float32)

train_ds = FaceAgeDS(train_df, train=True)
val_ds   = FaceAgeDS(val_df,   train=False)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ── CELL 5: Model — MobileNetV2 + Age Head ───────────────────
class AgeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),  nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x).squeeze(1)

model = AgeEstimator().to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 6: Training ─────────────────────────────────────────
criterion = nn.L1Loss()
EPOCHS_P1, EPOCHS_P2 = 5, 15
TOTAL = EPOCHS_P1 + EPOCHS_P2
best_mae, best_path = float("inf"), "/kaggle/working/age_best.pt"
history = {"tr": [], "vl": []}

def run(loader, train):
    model.train() if train else model.eval()
    all_p, all_t = [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for imgs, ages in loader:
            imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
            p = model(imgs)
            if train:
                opt.zero_grad(); criterion(p, ages).backward(); opt.step()
            all_p += p.detach().cpu().tolist()
            all_t += ages.cpu().tolist()
    return mean_absolute_error(all_t, all_p)

# Phase 1: head only
for p in model.features.parameters(): p.requires_grad = False
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P1)

for ep in range(1, TOTAL+1):
    if ep == EPOCHS_P1 + 1:
        print("\n--- Phase 2: full fine-tune ---")
        for p in model.features.parameters(): p.requires_grad = True
        opt = optim.AdamW(model.parameters(), lr=5e-5)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_P2, eta_min=1e-6)
    tr = run(train_loader, True)
    vl = run(val_loader,   False)
    sch.step()
    history["tr"].append(tr); history["vl"].append(vl)
    tag = ""
    if vl < best_mae:
        best_mae = vl; torch.save(model.state_dict(), best_path); tag = " ← BEST"
    print(f"Ep {ep:2d}/{TOTAL} | Train MAE {tr:.2f}y | Val MAE {vl:.2f}y{tag}")

# Plot
plt.figure(figsize=(8,4))
plt.plot(history["tr"], label="Train MAE"); plt.plot(history["vl"], label="Val MAE")
plt.axvline(EPOCHS_P1-1, color="red", linestyle="--", alpha=0.5, label="Phase 2")
plt.xlabel("Epoch"); plt.ylabel("MAE (years)")
plt.title(f"Training — Best Val MAE: {best_mae:.2f}y")
plt.legend(); plt.savefig("/kaggle/working/training.png"); plt.show()

# ── CELL 7: Indian-only evaluation ───────────────────────────
model.load_state_dict(torch.load(best_path)); model.eval()
test_indian = df[df.is_indian].sample(min(300, df.is_indian.sum()), random_state=99)
actual, pred = [], []
with torch.no_grad():
    for _, r in test_indian.iterrows():
        img = VAL_TF(Image.open(r.path).convert("RGB")).unsqueeze(0).to(DEVICE)
        pred.append(model(img).item()); actual.append(r.age)
indian_mae = mean_absolute_error(actual, pred)
print(f"\nIndian face MAE: {indian_mae:.2f} years")
print(f"Bias: {np.mean(pred)-np.mean(actual):+.1f}y (+ means overestimates)")

plt.figure(figsize=(7,6))
plt.scatter(actual, pred, alpha=0.4, color="#16a34a")
plt.plot([0,90],[0,90],"r--",label="Perfect")
plt.xlabel("Actual Age"); plt.ylabel("Predicted")
plt.title(f"Indian Faces: MAE={indian_mae:.1f}y, Bias={np.mean(pred)-np.mean(actual):+.1f}y")
plt.legend(); plt.savefig("/kaggle/working/indian_eval.png"); plt.show()

# ── CELL 8: Export ONNX ──────────────────────────────────────
model.load_state_dict(torch.load(best_path)); model.eval()
dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
onnx_path = "/kaggle/working/age_estimator_indian.onnx"

torch.onnx.export(
    model, dummy, onnx_path,
    opset_version=12,
    input_names=["face_crop"],
    output_names=["age"],
    dynamic_axes={"face_crop": {0: "batch"}, "age": {0: "batch"}},
)

import onnxruntime as ort, numpy as np
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test = np.random.randn(1, 3, 224, 224).astype(np.float32)
out  = sess.run(["age"], {"face_crop": test})[0]
print(f"ONNX verified! Output: {out[0]:.1f}y  |  Size: {os.path.getsize(onnx_path)/1e6:.1f} MB")
print(f"\nDownload from: /kaggle/working/age_estimator_indian.onnx")
print("Copy to: models/age_estimator_indian.onnx  in your project")
