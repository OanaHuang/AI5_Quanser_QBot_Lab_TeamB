#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following (CNN-Adaptive)-------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from hal.content.qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
from qlabs_setup import setup

# CNN-related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────
#  CNN Model Definition (must match train_cnn.py)
# ─────────────────────────────────────────────
LABELS      = sorted(["crossroad", "curve", "out_route", "straight", "t_junction"])
NUM_CLASSES = len(LABELS)
label2idx   = {l: i for i, l in enumerate(LABELS)}
idx2label   = {i: l for l, i in label2idx.items()}
CNN_MODEL_PATH = "test_model.pth"
# RTX 5060 (sm_120) is incompatible with the current PyTorch CUDA version; force CPU inference.
# Switch back to "cuda" after upgrading to a PyTorch nightly build that supports sm_120.
CNN_DEVICE     = torch.device("cpu")

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
        self.block1 = ConvBlock(3,   32,  dropout=0.1)
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

# CNN preprocessing transform (consistent with eval_tf in train_cnn.py)
cnn_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─────────────────────────────────────────────
#  Scene Parameter Table
#
#  col_bias  : Offset added to blob col.
#              Negative → col decreases → error=160-col increases → forces left turn.
#              Positive → forces right turn.
#              Unified left-turn strategy at intersections: col_bias = -70
#
#  spd_factor: Multiplied by the forSpd returned by line2SpdMap.
#              Reduce speed at intersections to allow enough time for turning.
#
#  kP / kD   : Steering gains passed to line2SpdMap.
#              Use large kP at intersections to quickly follow the left-turn line;
#              use small kD to avoid derivative saturation.
# ─────────────────────────────────────────────
SCENE_PARAMS = {
    # label        kP     kD    col_bias  spd_factor
    "straight":  {"kP": 0.35, "kD": 0.08, "col_bias":   0, "spd": 1.00},
    "curve":     {"kP": 0.55, "kD": 0.15, "col_bias":   0, "spd": 0.85},
    "crossroad": {"kP": 0.65, "kD": 0.05, "col_bias": -70, "spd": 0.55},
    "t_junction":{"kP": 0.65, "kD": 0.05, "col_bias": -70, "spd": 0.55},
    "out_route": {"kP": 0.15, "kD": 0.02, "col_bias":   0, "spd": 0.30},
}
_DEFAULT = {"kP": 0.40, "kD": 0.10, "col_bias": 0, "spd": 1.0}
CNN_CONF_THRESHOLD = 0.60   # Fall back to default values when confidence is below this threshold
CNN_INFER_EVERY    = 30     # Run inference every N frames (30 frames ≈ 0.5s @ 60Hz)

# ─────────────────────────────────────────────
#  CNN Inference Helper Functions
# ─────────────────────────────────────────────
def load_cnn_model(model_path: str) -> RoadCNN:
    """Load a trained RoadCNN and return the model in eval mode."""
    model = RoadCNN(NUM_CLASSES).to(CNN_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=CNN_DEVICE))
    model.eval()
    print(f"[CNN] Model loaded from {model_path}  (device={CNN_DEVICE})")
    return model

def infer_scene(model: RoadCNN, gray_frame: np.ndarray,
                conf_threshold: float = CNN_CONF_THRESHOLD):
    """
    Input:  grayscale numpy image (H×W or H×W×1)
    Output: (label_str, confidence, params_dict)
            params_dict contains kP, kD, col_bias, spd
    """
    if gray_frame.ndim == 2:
        rgb = np.stack([gray_frame] * 3, axis=-1)
    else:
        rgb = np.repeat(gray_frame, 3, axis=-1)
    pil_img = Image.fromarray(rgb.astype(np.uint8))

    tensor = cnn_tf(pil_img).unsqueeze(0).to(CNN_DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    label    = idx2label[pred_idx]
    conf     = float(probs[pred_idx])

    if conf >= conf_threshold:
        params = SCENE_PARAMS.get(label, _DEFAULT)
    else:
        label  = "low_conf"
        params = _DEFAULT

    return label, conf, params


# ─────────────────────────────────────────────
#  Section A - Setup
# ─────────────────────────────────────────────
hQBot = setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# CNN runtime state
cnn_model      = load_cnn_model(CNN_MODEL_PATH)
current_label  = "straight"
current_conf   = 1.0
current_params = SCENE_PARAMS["straight"]

try:
    # ─────────────────────────────────────────
    #  Section B - Initialization
    # ─────────────────────────────────────────
    myQBot      = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam     = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard    = Keyboard()
    vision      = QBPVision()
    probe       = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True,  scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[50,  320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    print("[CNN] Initial scene:", current_label,
          f"  kP={current_params['kP']:.3f}  kD={current_params['kD']:.3f}"
          f"  col_bias={current_params['col_bias']}  spd={current_params['spd']:.2f}")

    # Initial keyboard state (prevents undefined variables when newkeyboard=False on the first frame)
    lineFollow     = False
    keyboardComand = np.zeros(2)

    # ─────────────────────────────────────────
    #  Main Loop
    # ─────────────────────────────────────────
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard Driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm            = keyboard.k_space
                lineFollow     = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # Section C - Toggle line following
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # QBot Hardware
            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands,
            )

            if newHIL:
                timeHIL    = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown += 1

                    # ─────────────────────────────────────
                    #  Section D.1 - Undistort & Resize
                    # ─────────────────────────────────────
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))  # (H=200, W=320)

                    # ─────────────────────────────────────
                    #  Section D.CNN - Scene Classification
                    #  Run inference every CNN_INFER_EVERY frames to avoid impacting control frequency
                    # ─────────────────────────────────────
                    if counterDown % CNN_INFER_EVERY == 1:
                        new_label, new_conf, new_params = infer_scene(
                            cnn_model, gray_sm, CNN_CONF_THRESHOLD
                        )
                        if new_label != current_label:
                            print(f"[CNN] Scene: {current_label} → {new_label}"
                                  f"  conf={new_conf:.2f}"
                                  f"  kP={new_params['kP']:.3f}"
                                  f"  kD={new_params['kD']:.3f}"
                                  f"  col_bias={new_params['col_bias']}"
                                  f"  spd={new_params['spd']:.2f}")
                        current_label  = new_label
                        current_conf   = new_conf
                        current_params = new_params

                    # ─────────────────────────────────────
                    #  Section D.2 - Threshold & Blob
                    # ─────────────────────────────────────
                    rowStart, rowEnd         = 50, 100
                    minThreshold, maxThreshold = 100, 255

                    subImage = gray_sm[rowStart:rowEnd, :]
                    binary   = np.zeros_like(subImage)
                    height, width = subImage.shape

                    for i in range(height):
                        for j in range(width):
                            if minThreshold < subImage[i, j] < maxThreshold:
                                binary[i, j] = 255

                    connectivity = 8
                    min_pixels, max_pixels = 500, 2000
                    col, row, area = vision.image_find_objects(
                        binary, connectivity, min_pixels, max_pixels
                    )

                    # ─────────────────────────────────────
                    #  Section D.3 - Adaptive Speed Command
                    #
                    #  col_bias: shifts col at intersections to force a left turn
                    #    error = 160 - col + offset
                    #    col_bias < 0 → col decreases → error increases → left turn
                    #
                    #  spd_factor: reduces speed at intersections
                    # ─────────────────────────────────────
                    kP       = current_params["kP"]
                    kD       = current_params["kD"]
                    col_bias = current_params["col_bias"]
                    spd_fac  = current_params["spd"]

                    col_adj = (col + col_bias) if col is not None else col
                    forSpd, turnSpd = line2SpdMap.send((col_adj, kP, kD))
                    forSpd *= spd_fac

                if counterDown % 4 == 0:
                    probe.send(name='Raw Image',    imageData=gray_sm)
                    probe.send(name='Binary Image', imageData=binary)

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()