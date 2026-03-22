import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ----------------------- Config -----------------------
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10
SAVE_PATH   = "test_model_grey.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset Paths -------------------
IMG_DIR     = "images"
TRAIN_CSV   = "train.csv"
VAL_CSV     = "val.csv"
TEST_CSV    = "test.csv"
LABELS      = sorted(["crossroad", "curve", "out_route", "straight", "t_junction"])
NUM_CLASSES = len(LABELS)
label2idx   = {l: i for i, l in enumerate(LABELS)}
idx2label   = {i: l for l, i in label2idx.items()}

# ----------------------- Dataset ----------------------
class RoadDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        # Load as grayscale ("L" mode = single channel)
        image = Image.open(self.img_dir / row["image_name"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label2idx[row["label"]]

    def get_labels(self):
        return [label2idx[l] for l in self.df["label"]]

# -------------------- Transforms ----------------------
# Grayscale: normalize with single mean/std
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# -------------- Class Weight / Sampler ----------------
def compute_class_weights(csv_path: str) -> torch.Tensor:
    df     = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    total  = len(df)
    weights = torch.zeros(NUM_CLASSES)
    for label, idx in label2idx.items():
        weights[idx] = total / (NUM_CLASSES * counts[label])
    return weights.to(DEVICE)

def make_weighted_sampler(dataset: RoadDataset) -> WeightedRandomSampler:
    labels       = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )
    return sampler

# ------------------- DataLoaders ----------------------
def make_train_loader(csv_path: str) -> DataLoader:
    ds      = RoadDataset(csv_path, IMG_DIR, train_tf)
    sampler = make_weighted_sampler(ds)
    return DataLoader(
        ds,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = False,
    )

def make_eval_loader(csv_path: str) -> DataLoader:
    ds = RoadDataset(csv_path, IMG_DIR, eval_tf)
    return DataLoader(
        ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

# ----------------------- Model ------------------------
# Architecture reference: https://fastai.github.io/fastbook2e/resnet.html
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.pool     = nn.MaxPool2d(2, 2)
        self.drop     = nn.Dropout2d(dropout)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        sc  = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.drop(self.pool(F.relu(out + sc)))

class RoadCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # in_ch=1 for single-channel grayscale input
        self.block1 = ConvBlock(1,   32,  dropout=0.1)
        self.block2 = ConvBlock(32,  64,  dropout=0.2)
        self.block3 = ConvBlock(64,  128, dropout=0.3)
        self.block4 = ConvBlock(128, 256, dropout=0.3)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(self.gap(x))

# ------------- Training / Evaluation Loop -------------
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False,
                                   desc="train" if training else "eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

# ------------------------- Main -----------------------
if __name__ == "__main__":

    print(f"Device : {DEVICE}")

    model = RoadCNN(NUM_CLASSES).to(DEVICE)
    print(f"Classes: {LABELS}")

    train_loader = make_train_loader(TRAIN_CSV)
    val_loader   = make_eval_loader(VAL_CSV)
    test_loader  = make_eval_loader(TEST_CSV)

    class_weights = compute_class_weights(TRAIN_CSV)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = LR,
        steps_per_epoch = len(train_loader),
        epochs          = EPOCHS,
        pct_start       = 0.1,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, patience_counter = 0.0, 0

    print("\n" + "=" * 65)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, scheduler, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion,
                                    training=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            patience_counter = 0
            flag = " <- best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.4f} | "
              f"{time.time() - t0:.1f}s{flag}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Evaluate on test set using best checkpoint
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    label_names = [idx2label[i] for i in range(NUM_CLASSES)]

    print("Test result:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Loss / Accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"train_{key}"], label="Train")
        ax.plot(history[f"val_{key}"],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

'''
def predict(img_path: str, model_path: str = SAVE_PATH) -> str:
    """Single-image inference helper."""
    m = RoadCNN(NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    m.eval()
    image  = Image.open(img_path).convert("L")
    tensor = eval_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(m(tensor), dim=1).cpu().numpy()[0]
    top  = idx2label[int(probs.argmax())]
    conf = {idx2label[i]: round(float(p), 4) for i, p in enumerate(probs)}
    print(f"Predicted label : {top}")
    print(f"Class confidence: {conf}")
    return top
'''

