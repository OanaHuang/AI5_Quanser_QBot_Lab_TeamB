#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------  + CNN-Adaptive PID Control  ------------------------#
#-----------------------------------------------------------------------------#

# ============================================================
#  CNN Architecture (matches baseline_cnn_checkpoint.pth)
#
#  Input : RGB image, any spatial size (AdaptiveAvgPool handles it)
#          In practice we feed the 50×320 sub-image repeated to 3 ch.
#
#  features.N.block:
#    .0  Conv2d   (3→32→64→128→256, kernel 3×3, pad 1)
#    .1  BatchNorm2d
#    .2  ReLU
#    .3  MaxPool2d(2)
#
#  pool        : AdaptiveAvgPool2d(1)  → (B, 256, 1, 1)
#
#  classifier  :
#    .0  Flatten
#    .1  Linear(256 → 128)
#    .2  ReLU
#    .3  Dropout(0.5)
#    .4  Linear(128 → 5)
#
#  Class labels (index → road condition):
#    0  Straight
#    1  Slight-Left  curve
#    2  Slight-Right curve
#    3  Sharp-Left   curve
#    4  Sharp-Right  curve
# ============================================================

# ---------- Standard imports ----------
import time
import numpy as np
import cv2
from collections import deque

# ---------- PyTorch ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Quanser imports ----------
from pal.products.qbot_platform import (
    QBotPlatformDriver, Keyboard,
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar,
)
from hal.content.qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
from qlabs_setup import setup


# ─────────────────────────────────────────────────────────────
#  Section 0 – CNN Model Definition
# ─────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Single feature block: Conv → BN → ReLU → MaxPool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # block.0
            nn.BatchNorm2d(out_ch),                                            # block.1
            nn.ReLU(inplace=True),                                             # block.2
            nn.MaxPool2d(kernel_size=2),                                       # block.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    4-block CNN with global average pooling + 2-layer MLP head.
    Exactly matches the key layout in baseline_cnn_checkpoint.pth.
    """
    NUM_CLASSES = 5

    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([
            _ConvBlock(3,   32),   # features.0
            _ConvBlock(32,  64),   # features.1
            _ConvBlock(64,  128),  # features.2
            _ConvBlock(128, 256),  # features.3
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),           # classifier.0
            nn.Linear(256, 128),    # classifier.1
            nn.ReLU(inplace=True),  # classifier.2
            nn.Dropout(p=0.5),      # classifier.3
            nn.Linear(128, self.NUM_CLASSES),  # classifier.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.features:
            x = block(x)
        x = self.pool(x)
        return self.classifier(x)


def load_cnn(checkpoint_path: str, device: torch.device) -> BaselineCNN:
    """Load model weights from checkpoint; returns model in eval mode."""
    model = BaselineCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both raw state-dict and {'model_state_dict': ...} wrappers
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def classify_image(model: BaselineCNN,
                   gray_crop: np.ndarray,
                   device: torch.device) -> int:
    """
    Preprocess a grayscale numpy crop and return the predicted class index.

    gray_crop : H×W uint8 grayscale image (already undistorted & cropped)
    """
    # Convert grayscale to 3-channel by repeating the single channel
    rgb = np.stack([gray_crop, gray_crop, gray_crop], axis=2)  # H×W×3

    # Normalise to [0, 1] then apply ImageNet-style mean/std
    tensor = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std                             # H×W×3

    # HWC → CHW → BCHW
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(tensor)                    # (1, 5)
    return int(logits.argmax(dim=1).item())


# ─────────────────────────────────────────────────────────────
#  Section 1 – Per-Class PID / Speed Parameters
#
#  Tune these to match your physical track.
#
#  'fwd_spd'    : target forward speed sent to line_to_speed_map  [m/s]
#  'steer_gain' : proportional steering gain  (≈ kp)
#  'kd'         : derivative damping gain     (applied to Δerror)
#
#  The final speed command is computed as:
#     forSpd, turnSpd = line2SpdMap.send((col, fwd_spd, steer_gain))
#  with an additional kd*d_error correction added to turnSpd.
# ─────────────────────────────────────────────────────────────

CLASS_PARAMS = {
    #  idx  name             fwd_spd  steer_gain  kd
    0: dict(name='Straight',     fwd_spd=0.45, steer_gain=0.08, kd=0.015),
    1: dict(name='SlightLeft',   fwd_spd=0.30, steer_gain=0.14, kd=0.030),
    2: dict(name='SlightRight',  fwd_spd=0.30, steer_gain=0.14, kd=0.030),
    3: dict(name='SharpLeft',    fwd_spd=0.18, steer_gain=0.25, kd=0.055),
    4: dict(name='SharpRight',   fwd_spd=0.18, steer_gain=0.25, kd=0.055),
}

# Image column index of the line centre target (centre of 320-px wide image)
IMAGE_CENTRE = 160.0

# How many recent CNN predictions to use for majority-vote smoothing
SMOOTH_WINDOW = 5

# Run CNN inference every N HIL frames (60 Hz / 2 = 30 Hz classification)
CNN_STRIDE = 2

# Path to the model checkpoint (adjust if needed)
CHECKPOINT_PATH = 'baseline_cnn_checkpoint.pth'


# ─────────────────────────────────────────────────────────────
#  Section A – Setup
# ─────────────────────────────────────────────────────────────

hQBot = setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros(2, dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1 / 60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0.0, 0.0
startTime = time.time()


def elapsed_time():
    return time.time() - startTime


timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# ── Load the CNN model ──────────────────────────────────────
# Force CPU: RTX 5060 (sm_120) is not yet supported by the bundled
# PyTorch CUDA kernels (max sm_90).  CPU inference is fast enough
# at 30 Hz on a 50×320 crop.
device = torch.device('cpu')
print(f'[CNN] Loading model from "{CHECKPOINT_PATH}" on {device} …')
cnn_model = load_cnn(CHECKPOINT_PATH, device)
print('[CNN] Model loaded successfully.')

# ── Runtime CNN state ───────────────────────────────────────
cnn_class        = 0                              # current predicted class
cnn_label        = CLASS_PARAMS[0]['name']        # human-readable label
cnn_vote_buf     = deque([0] * SMOOTH_WINDOW,     # smoothing ring buffer
                          maxlen=SMOOTH_WINDOW)
active_params    = CLASS_PARAMS[0].copy()         # live PID params
prev_error       = 0.0                            # for derivative term

# ── Fallback defaults (used when no blob is detected) ───────
DEFAULT_FWD_SPD    = 0.0
DEFAULT_STEER_GAIN = 0.08
DEFAULT_KD         = 0.015

# ── Pre-initialise keyboard state ───────────────────────────
# keyboardCommand holds [forward, turn] from the keyboard driver.
# Initialise here so the variable always exists before the first
# loop iteration, even if keyboard.read() returns False.
keyboardCommand = np.zeros(2, dtype=np.float64)

try:
    # ─────────────────────────────────────────────────────────
    #  Section B – Initialization
    # ─────────────────────────────────────────────────────────
    myQBot   = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam  = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision   = QBPVision()
    probe    = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True,  scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[ 50, 320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    # Build the PD speed mapper (generator)
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)

    startTime = time.time()
    time.sleep(0.5)
    lineFollow = False   # toggled by keyboard key 7

    # ─────────────────────────────────────────────────────────
    #  Main Loop
    # ─────────────────────────────────────────────────────────
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # ── Keyboard input ────────────────────────────────
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm        = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardCommand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # ── Section C – Toggle line following ─────────────
            if not lineFollow:
                # Manual drive
                commands = np.array([keyboardCommand[0], keyboardCommand[1]],
                                    dtype=np.float64)
            else:
                # Autonomous line following (overwritten below after processing)
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # ── QBot HIL read/write ───────────────────────────
            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands,
            )

            if newHIL:
                timeHIL   = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown += 1

                    # ─────────────────────────────────────────
                    #  Section D – Image Processing
                    # ─────────────────────────────────────────

                    # D.1 – Undistort and resize
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm     = cv2.resize(undistorted, (320, 200))

                    rowStart, rowEnd = 50, 100
                    subImage = gray_sm[rowStart:rowEnd, :]   # 50×320 grayscale

                    # D.2 – Threshold → binary mask
                    maxThreshold, minThreshold = 255, 100
                    binary = np.zeros_like(subImage)
                    h, w   = subImage.shape
                    for i in range(h):
                        for j in range(w):
                            if minThreshold < subImage[i, j] < maxThreshold:
                                binary[i, j] = 255

                    # D.3 – Blob detection (Connected Component Labelling)
                    connectivity = 8
                    min_pixels, max_pixels = 500, 2000
                    col, row, area = vision.image_find_objects(
                        binary, connectivity, min_pixels, max_pixels
                    )

                    # ─────────────────────────────────────────
                    #  Section E – CNN Classification
                    #  (runs every CNN_STRIDE frames)
                    # ─────────────────────────────────────────
                    if lineFollow and (counterDown % CNN_STRIDE == 0):
                        raw_class = classify_image(cnn_model, subImage, device)

                        # Majority-vote smoothing over the last SMOOTH_WINDOW frames
                        cnn_vote_buf.append(raw_class)
                        votes     = list(cnn_vote_buf)
                        cnn_class = max(set(votes), key=votes.count)
                        cnn_label = CLASS_PARAMS[cnn_class]['name']

                        # Update active PID parameters
                        active_params = CLASS_PARAMS[cnn_class]

                        print(
                            f'[CNN] class={cnn_class} ({cnn_label}) | '
                            f'fwd={active_params["fwd_spd"]:.2f}  '
                            f'kp={active_params["steer_gain"]:.3f}  '
                            f'kd={active_params["kd"]:.3f}'
                        )

                    # ─────────────────────────────────────────
                    #  Section F – Speed Command (CNN-Adaptive PD)
                    # ─────────────────────────────────────────
                    if lineFollow and col is not None:
                        # Retrieve class-specific gains
                        fwd_spd    = active_params['fwd_spd']
                        steer_gain = active_params['steer_gain']
                        kd         = active_params['kd']

                        # Proportional error: positive = line is to the right
                        error   = (col - IMAGE_CENTRE) / IMAGE_CENTRE   # normalised [-1, 1]
                        d_error = error - prev_error
                        prev_error = error

                        # Base forward/turn speeds from the Quanser PD mapper
                        forSpd, turnSpd = line2SpdMap.send((col, fwd_spd, steer_gain))

                        # Add derivative correction to dampen oscillations
                        turnSpd += kd * d_error

                    elif not lineFollow:
                        # Reset derivative memory when switching to manual
                        prev_error = 0.0

                    # Probe display (every 4 frames)
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