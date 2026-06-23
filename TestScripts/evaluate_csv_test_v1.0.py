import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 32
NUM_WORKERS = 0
CHECKPOINT_NAME = "baseline_cnn_checkpoint.pth"
CONF_THRESHOLD = 0.60
DEFAULT_CLASS = "straight"


# =========================================================
# Model (single-channel grayscale)
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineSceneCNNGray(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def adapt_checkpoint_to_gray(state_dict: dict) -> dict:
    adapted = {}
    for k, v in state_dict.items():
        if k == "features.0.block.0.weight":
            if v.ndim == 4 and v.shape[1] == 3:
                adapted[k] = v.mean(dim=1, keepdim=True)
            else:
                adapted[k] = v
        else:
            adapted[k] = v
    return adapted


# =========================================================
# Dataset
# =========================================================
class QbotSceneDatasetGray(Dataset):
    def __init__(self, img_dir: str, csv_path: str, class_to_idx: dict, img_size):
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.img_size = tuple(img_size)

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        cols_lower = [c.lower() for c in self.df.columns]
        if "image" in cols_lower and "label" in cols_lower:
            self.image_col = self.df.columns[cols_lower.index("image")]
            self.label_col = self.df.columns[cols_lower.index("label")]
        else:
            if len(self.df.columns) < 2:
                raise ValueError("CSV must contain at least two columns: image and label")
            self.image_col = self.df.columns[0]
            self.label_col = self.df.columns[1]

    def __len__(self) -> int:
        return len(self.df)

    def preprocess_image(self, img_gray: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img_gray, self.img_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        chw = np.expand_dims(normalized, axis=0)  # 1 x H x W
        return chw

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = str(row[self.image_col]).strip()
        label_value = row[self.label_col]

        img_path = os.path.join(self.img_dir, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        image_tensor = torch.tensor(self.preprocess_image(img), dtype=torch.float32)

        if isinstance(label_value, str):
            label_key = label_value.strip()
            if label_key not in self.class_to_idx:
                raise ValueError(f"Unknown label '{label_key}' in CSV")
            class_index = self.class_to_idx[label_key]
        else:
            class_index = int(label_value)

        label_tensor = torch.tensor(class_index, dtype=torch.long)
        return image_tensor, label_tensor, image_name


# =========================================================
# Evaluation
# =========================================================
def evaluate_model(model, dataloader, device, idx_to_label):
    model.eval()

    total_samples = 0
    total_correct_raw = 0
    total_correct_with_fallback = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            total_correct_raw += (preds == labels).sum().item()

            fallback_preds = preds.clone()
            low_conf_mask = confs < CONF_THRESHOLD
            fallback_preds[low_conf_mask] = idx_to_label.inverse_default_idx
            total_correct_with_fallback += (fallback_preds == labels).sum().item()

            total_samples += images.size(0)

    raw_acc = total_correct_raw / max(total_samples, 1)
    fallback_acc = total_correct_with_fallback / max(total_samples, 1)
    return raw_acc, fallback_acc


def print_sample_predictions(model, dataloader, device, idx_to_label, default_idx, max_samples=20):
    model.eval()
    printed = 0

    with torch.no_grad():
        for images, labels, image_names in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                true_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                conf_val = float(confs[i].item())

                final_idx = pred_idx if conf_val >= CONF_THRESHOLD else default_idx

                print(
                    f"Image: {image_names[i]} | "
                    f"True: {idx_to_label[true_idx]} | "
                    f"RawPred: {idx_to_label[pred_idx]} | "
                    f"FinalPred: {idx_to_label[final_idx]} | "
                    f"Conf: {conf_val:.4f}"
                )

                printed += 1
                if printed >= max_samples:
                    return


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "smallraw")
    csv_path = os.path.join(base_dir, "coords.csv")
    checkpoint_path = os.path.join(base_dir, CHECKPOINT_NAME)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Expected a checkpoint dict with key 'model_state_dict'")

    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_label = {v: k for k, v in class_to_idx.items()}

    default_idx = class_to_idx[DEFAULT_CLASS]

    # small helper so evaluate_model can access default class idx
    idx_to_label.inverse_default_idx = default_idx

    config = ckpt.get("config", {})
    img_size = tuple(config.get("img_size", (128, 128)))
    num_classes = int(config.get("num_classes", len(classes)))

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Classes: {classes}")
    print(f"Class to idx: {class_to_idx}")
    print(f"Image size: {img_size}")
    print(f"Num classes: {num_classes}")
    print("Evaluation input mode: SINGLE-CHANNEL GRAYSCALE")

    dataset = QbotSceneDatasetGray(
        img_dir=img_dir,
        csv_path=csv_path,
        class_to_idx=class_to_idx,
        img_size=img_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = BaselineSceneCNNGray(num_classes=num_classes).to(device)
    state_dict = adapt_checkpoint_to_gray(ckpt["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    raw_acc, fallback_acc = evaluate_model(model, dataloader, device, idx_to_label)

    print(f"Dataset size: {len(dataset)}")
    print(f"Raw accuracy: {raw_acc:.4f}")
    print(f"Accuracy with low-conf fallback -> '{DEFAULT_CLASS}': {fallback_acc:.4f}")

    print("\nSample predictions:")
    print_sample_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        idx_to_label=idx_to_label,
        default_idx=default_idx,
        max_samples=20,
    )


if __name__ == "__main__":
    main()