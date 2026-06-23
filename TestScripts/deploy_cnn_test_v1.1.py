import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, QBotPlatformCSICamera
from pal.utilities.probe import Probe
from hal.content.qbot_platform_functions import QBPVision
from quanser.hardware import HILError


# =========================================================
# Config
# =========================================================
CHECKPOINT_NAME = "baseline_cnn_checkpoint.pth"

CNN_INFER_EVERY = 10
CLASS_HISTORY = 5
CONF_THRESHOLD = 0.60

DEFAULT_CLASS = "straight"

MAX_STEER = 0.5
MAX_INTEGRAL = 300.0

ROI_ROW_START = 50
ROI_ROW_END = 120
BINARY_THRESH = 110
MIN_PIXELS = 350
MAX_PIXELS = 12000
BLOB_CONNECTIVITY = 8

SHOW_DEBUG_EVERY = 4

# Scene-adaptive PID params
SCENE_PARAMS = {
    "straight": {
        "kp": 0.0065,
        "ki": 0.0000,
        "kd": 0.0018,
        "speed": 0.14,
        "bias": 0.0,
    },
    "curve": {
        "kp": 0.0090,
        "ki": 0.0000,
        "kd": 0.0025,
        "speed": 0.10,
        "bias": 0.0,
    },
    "crossroad": {
        "kp": 0.0105,
        "ki": 0.0000,
        "kd": 0.0020,
        "speed": 0.08,
        "bias": -40.0,   # mild left-turn prior
    },
    "t_junction": {
        "kp": 0.0105,
        "ki": 0.0000,
        "kd": 0.0020,
        "speed": 0.08,
        "bias": -55.0,   # stronger left-turn prior
    },
    "out_route": {
        "kp": 0.0050,
        "ki": 0.0000,
        "kd": 0.0015,
        "speed": 0.05,
        "bias": 0.0,
    },
}


# =========================================================
# CNN model components
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

        # fix
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # resnet pth classifier.1 和 classifier.4
        self.classifier = nn.Sequential(
            nn.Identity(),                # Index 0
            nn.Linear(256, 128),          # Index 1
            nn.ReLU(inplace=True),        # Index 2
            nn.Dropout(p=0.5),            # Index 3
            nn.Linear(128, num_classes),  # Index 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def adapt_checkpoint_to_gray(state_dict: dict) -> dict:
    """
    Allow loading old 3-channel checkpoints into the new 1-channel model
    by averaging the first conv weights across RGB channels.
    """
    adapted = {}
    for k, v in state_dict.items():
        if k == "features.0.block.0.weight":
            # old shape: [32, 3, 3, 3] -> new shape: [32, 1, 3, 3]
            if v.ndim == 4 and v.shape[1] == 3:
                adapted[k] = v.mean(dim=1, keepdim=True)
            else:
                adapted[k] = v
        else:
            adapted[k] = v
    return adapted


# =========================================================
# PID controller
# =========================================================
class PIDController:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float, kp: float, ki: float, kd: float) -> float:
        if dt <= 1e-6:
            dt = 1e-3

        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -MAX_INTEGRAL, MAX_INTEGRAL))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        steer = kp * error + ki * self.integral + kd * derivative
        return float(np.clip(steer, -MAX_STEER, MAX_STEER))


# =========================================================
# CNN preprocessing / inference
# =========================================================
def preprocess_for_cnn(gray_frame: np.ndarray, img_size) -> torch.Tensor:
    if gray_frame is None or gray_frame.size == 0:
        raise ValueError("Empty frame received for CNN preprocessing")

    if gray_frame.ndim == 3:
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray_frame, tuple(img_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5  # match torchvision Normalize(mean=0.5, std=0.5)
    chw = np.expand_dims(normalized, axis=0)   # 1 x H x W
    chw = np.expand_dims(chw, axis=0)          # B x 1 x H x W
    return torch.tensor(chw, dtype=torch.float32)


def classify_scene(gray_frame: np.ndarray, model, device, img_size, idx_to_label):
    input_tensor = preprocess_for_cnn(gray_frame, img_size).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_idx = int(pred.item())
    conf_val = float(conf.item())
    label = idx_to_label.get(pred_idx, DEFAULT_CLASS)

    if conf_val < CONF_THRESHOLD:
        label = DEFAULT_CLASS

    return label, conf_val


def get_stable_class(history: deque, default_class: str) -> str:
    if not history:
        return default_class
    counter = Counter(history)
    return counter.most_common(1)[0][0]


# =========================================================
# Blob-based line detection with visual toolchain
# =========================================================
def threshold_and_find_blob(gray_sm: np.ndarray, vision: QBPVision):
    h, w = gray_sm.shape
    row_start = max(0, min(ROI_ROW_START, h - 1))
    row_end = max(row_start + 1, min(ROI_ROW_END, h))
    roi = gray_sm[row_start:row_end, :]

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, binary = cv2.threshold(blurred, BINARY_THRESH, 255, cv2.THRESH_BINARY)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    col, row, area = vision.image_find_objects(
        binary,
        BLOB_CONNECTIVITY,
        MIN_PIXELS,
        MAX_PIXELS,
    )

    debug_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    if col is None or row is None or area is None:
        return None, binary, debug_roi, None

    img_center_x = w / 2.0
    error_px = img_center_x - float(col)

    cv2.circle(debug_roi, (int(col), int(row)), 5, (255, 255, 255), -1)
    cv2.line(debug_roi, (int(img_center_x), 0), (int(img_center_x), roi.shape[0] - 1), (180, 180, 180), 1)

    blob_info = {
        "col": float(col),
        "row": float(row),
        "area": float(area),
        "row_start": row_start,
        "row_end": row_end,
    }
    return float(error_px), binary, debug_roi, blob_info


# =========================================================
# Main
# =========================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, CHECKPOINT_NAME)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise ValueError("Expected a checkpoint dict with key 'model_state_dict'")

    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_label = {v: k for k, v in class_to_idx.items()}

    config = ckpt.get("config", {})
    img_size = tuple(config.get("img_size", (128, 128)))
    num_classes = int(config.get("num_classes", len(classes)))

    # initialize
    model = BaselineSceneCNNGray(num_classes=num_classes).to(device)
    state_dict = adapt_checkpoint_to_gray(ckpt["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Classes: {classes}")
    print(f"Image size: {img_size}")

    qbot = None
    down_cam = None
    keyboard = None
    probe = None
    vision = None

    try:
        qbot = QBotPlatformDriver(mode=1, ip="localhost")
        down_cam = QBotPlatformCSICamera(frameRate=60.0, exposure=39.0, gain=17.0)
        keyboard = Keyboard()
        probe = Probe(ip="localhost")
        vision = QBPVision()

        probe.add_display(imageSize=[200, 320, 1], scaling=True, scalingFactor=2, name="Raw Gray")
        probe.add_display(imageSize=[ROI_ROW_END - ROI_ROW_START, 320, 1], scaling=False, scalingFactor=2, name="Binary ROI")

        pid = PIDController()
        class_history = deque(maxlen=CLASS_HISTORY)

        frame_count = 0
        stable_class = DEFAULT_CLASS
        current_conf = 1.0

        auto_mode = False
        arm = 0
        no_kill = True

        last_time = time.time()
        last_manual_cmd = np.zeros(2, dtype=np.float64)

        while no_kill:
            now = time.time()
            dt = now - last_time
            last_time = now

            if not probe.connected:
                probe.check_connection()

            new_key = keyboard.read()
            if new_key:
                arm = keyboard.k_space
                auto_mode = keyboard.k_7
                last_manual_cmd = np.array([keyboard.bodyCmd[0], keyboard.bodyCmd[1]], dtype=np.float64)
                if keyboard.k_u:
                    no_kill = False

            commands = last_manual_cmd.copy()

            new_hil = qbot.read_write_std(timestamp=time.time(), arm=arm, commands=commands)
            if not new_hil:
                continue

            new_down = down_cam.read()
            if not new_down:
                continue

            frame_count += 1
            undistorted = vision.df_camera_undistort(down_cam.imageData)
            gray_sm = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY) if undistorted.ndim == 3 else undistorted.copy()
            gray_sm = cv2.resize(gray_sm, (320, 200), interpolation=cv2.INTER_AREA)

            if frame_count % CNN_INFER_EVERY == 1:
                pred_label, conf_val = classify_scene(gray_sm, model, device, img_size, idx_to_label)
                class_history.append(pred_label)
                stable_class = get_stable_class(class_history, DEFAULT_CLASS)
                current_conf = conf_val

            params = SCENE_PARAMS.get(stable_class, SCENE_PARAMS[DEFAULT_CLASS])
            line_error, binary_roi, debug_roi, blob_info = threshold_and_find_blob(gray_sm, vision)

            if auto_mode:
                if line_error is None:
                    commands = np.array([0.04, 0.0], dtype=np.float64)
                    pid.reset()
                else:
                    steer_cmd = pid.update(line_error + params["bias"], dt, params["kp"], params["ki"], params["kd"])
                    commands = np.array([params["speed"], steer_cmd], dtype=np.float64)

                qbot.read_write_std(timestamp=time.time(), arm=arm, commands=commands)

            if frame_count % CNN_INFER_EVERY == 1:
                print(f"[Scene] {stable_class:>10s} | conf={current_conf:.3f} | speed={params['speed']:.3f}")

            if probe.connected and frame_count % SHOW_DEBUG_EVERY == 0:
                probe.send(name="Raw Gray", imageData=gray_sm)
                probe.send(name="Binary ROI", imageData=binary_roi)

    except KeyboardInterrupt:
        pass
    except HILError as h:
        print(h.get_error_message())
    finally:
        for obj in [down_cam, qbot, probe, keyboard]:
            if obj is not None: obj.terminate()

if __name__ == "__main__":
    main()