# -----------------------------------------------------------------------------#
# ------------------Skills Progression 1 - Task Automation---------------------#
# -----------------------------------------------------------------------------#
# ----------------------------Lab 3 - Line Following (CNN-Adaptive)-------------#
# -----------------------------------------------------------------------------#

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
#  CNN Model Definition
# ─────────────────────────────────────────────
LABELS = sorted(["crossroad", "curve", "out_route", "straight", "t_junction"])
NUM_CLASSES = len(LABELS)
label2idx = {l: i for i, l in enumerate(LABELS)}
idx2label = {i: l for l, i in label2idx.items()}
CNN_MODEL_PATH = "test_model.pth"
CNN_DEVICE = torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(dropout)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        sc = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.drop(self.pool(F.relu(out + sc)))


class RoadCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = ConvBlock(3, 32, dropout=0.1)
        self.block2 = ConvBlock(32, 64, dropout=0.2)
        self.block3 = ConvBlock(64, 128, dropout=0.3)
        self.block4 = ConvBlock(128, 256, dropout=0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
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


# CNN preprocessing transform
cnn_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─────────────────────────────────────────────
#  Scene Parameter Table
# ─────────────────────────────────────────────
SCENE_PARAMS = {
    "straight": {"kP": 0.4, "kD": 0.4, "col_bias": 0, "spd": 0.40},
    "curve": {"kP": 0.4, "kD": 0.4, "col_bias": 0, "spd": 0.40},
    "crossroad": {"kP": 2.5, "kD": 0.6, "col_bias": -200, "spd": -0.05},
    "t_junction": {"kP": 2.5, "kD": 0.6, "col_bias": -200, "spd": -0.05},
    "out_route": {"kP": 0.4, "kD": 0.4, "col_bias": 0, "spd": 0.4},
}
_DEFAULT = {"kP": 0.40, "kD": 0.40, "col_bias": 0, "spd": 0.3}
CNN_CONF_THRESHOLD = 0.50
CNN_INFER_EVERY = 30

# ─────────────────────────────────────────────
#  Junction State Machine — Tunable Parameters
# ─────────────────────────────────────────────
JUNC_STOP_DURATION = 0.5
JUNC_TURN_SPEED = 0.50  # Positive = Left Turn (Standard ROS/Quanser Kinematics)
JUNC_TURN_DURATION = (np.pi / 2) / JUNC_TURN_SPEED
JUNC_DRIVE_SPEED = 0.15
JUNC_MIN_DRIVE_TIME = 0.5
JUNC_COOLDOWN_FRAMES = 90

# Dynamic Speed Damping Parameters
TURN_SPD_MAX = 75.0
DAMPING_MAX = 0.60

JUNC_FOLLOW = "follow"
JUNC_STOPPING = "stopping"
JUNC_TURNING = "turning"
JUNC_DRIVING = "driving_through"


# ─────────────────────────────────────────────
#  CNN Inference Helper Functions
# ─────────────────────────────────────────────
def load_cnn_model(model_path: str) -> RoadCNN:
    model = RoadCNN(NUM_CLASSES).to(CNN_DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=CNN_DEVICE))
    model.eval()
    print(f"[CNN] Model loaded from {model_path}  (device={CNN_DEVICE})")
    return model


def infer_scene(model: RoadCNN, gray_frame: np.ndarray,
                conf_threshold: float = CNN_CONF_THRESHOLD):
    if gray_frame.ndim == 2:
        rgb = np.stack([gray_frame] * 3, axis=-1)
    else:
        rgb = np.repeat(gray_frame, 3, axis=-1)
    pil_img = Image.fromarray(rgb.astype(np.uint8))

    tensor = cnn_tf(pil_img).unsqueeze(0).to(CNN_DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    label = idx2label[pred_idx]
    conf = float(probs[pred_idx])

    if conf >= conf_threshold:
        params = SCENE_PARAMS.get(label, _DEFAULT)
    else:
        label = "low_conf"
        params = _DEFAULT

    return label, conf, params


# ─────────────────────────────────────────────
#  Section A - Setup
# ─────────────────────────────────────────────
hQBot = setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1 / 60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0

startTime = time.time()


def elapsed_time():
    return time.time() - startTime


timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

cnn_model = load_cnn_model(CNN_MODEL_PATH)
current_label = "straight"
current_conf = 1.0
current_params = SCENE_PARAMS["straight"]

# ── FSM Runtime Variables ────────────────────────────────────────────────────
junc_state = JUNC_FOLLOW
junc_phase_start = 0.0
junc_turn_sign = 1
junc_cooldown_count = 0

# Crossroad Sequence Logic: 1 (Left) -> -1 (Right) -> 0 (Straight)
turn_sequence = [1, -1, 0]
turn_index = 0

# Failsafe Recovery Variables
is_recovering = False
recovery_start_time = 0.0
last_valid_turn = 0.0
# ─────────────────────────────────────────────────────────────────────────────

try:
    # ─────────────────────────────────────────
    #  Section B - Initialization
    # ─────────────────────────────────────────
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision = QBPVision()
    probe = Probe(ip=ipHost)
    probe.add_display(imageSize=[200, 320, 1], scaling=True, scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[50, 320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    print("[CNN] Initial scene:", current_label)

    lineFollow = False
    keyboardComand = np.zeros(2)
    binary = np.zeros((50, 320), dtype=np.uint8)

    # ─────────────────────────────────────────
    #  Main Loop
    # ─────────────────────────────────────────
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # Motor Command Selection
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            elif junc_state == JUNC_STOPPING:
                commands = np.array([0.0, 0.0], dtype=np.float64)
            elif junc_state == JUNC_TURNING:
                commands = np.array([0.0, junc_turn_sign * JUNC_TURN_SPEED], dtype=np.float64)
            elif junc_state == JUNC_DRIVING:
                commands = np.array([JUNC_DRIVE_SPEED, 0.0], dtype=np.float64)
            else:
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            newHIL = myQBot.read_write_std(
                timestamp=time.time() - startTime,
                arm=arm,
                commands=commands,
            )

            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown += 1

                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))

                    # Optimized Threshold for White Line (Filters out gray floor)
                    rowStart, rowEnd = 50, 100
                    minThreshold, maxThreshold = 200, 255

                    subImage = gray_sm[rowStart:rowEnd, :]
                    binary = np.zeros_like(subImage)
                    h_sub, w_sub = subImage.shape

                    for i in range(h_sub):
                        for j in range(w_sub):
                            if minThreshold < subImage[i, j] < maxThreshold:
                                binary[i, j] = 255

                    connectivity = 8
                    min_pixels, max_pixels = 100, 15000
                    col, row, area = vision.image_find_objects(
                        binary, connectivity, min_pixels, max_pixels
                    )

                    # CNN Inference Trigger
                    if (junc_state == JUNC_FOLLOW and counterDown % CNN_INFER_EVERY == 1):
                        new_label, new_conf, new_params = infer_scene(
                            cnn_model, gray_sm, CNN_CONF_THRESHOLD
                        )
                        if new_label != current_label:
                            print(f"[CNN] Scene: {current_label} → {new_label} (conf={new_conf:.2f})")
                        current_label = new_label
                        current_conf = new_conf
                        current_params = new_params

                    if (junc_state == JUNC_DRIVING and counterDown % CNN_INFER_EVERY == 1):
                        new_label, new_conf, _ = infer_scene(
                            cnn_model, gray_sm, CNN_CONF_THRESHOLD
                        )
                        current_label = new_label
                        current_conf = new_conf

                    # ──────────────────────────────────────────────────────────
                    #  Section D.JUNC & D.3 - Combined Control & FSM Transitions
                    # ──────────────────────────────────────────────────────────
                    if junc_cooldown_count > 0:
                        junc_cooldown_count -= 1

                    now = elapsed_time()

                    # Highest Priority: Did we lose the line completely?
                    is_line_lost = (current_label == "out_route" or col is None or area == 0)

                    if junc_state == JUNC_FOLLOW:
                        is_junction = current_label in ("crossroad", "t_junction")

                        # 1. Failsafe Recovery Trigger
                        if is_line_lost:
                            if not is_recovering:
                                is_recovering = True
                                recovery_start_time = now
                                print(f"⚠️ WARNING: Line lost! Initiating reverse recovery...")

                            recovery_duration = now - recovery_start_time
                            if recovery_duration < 3.0:
                                forSpd = -0.10
                                turnSpd = float(np.clip(last_valid_turn, -0.3, 0.3))
                            else:
                                if counterDown % 30 == 0:
                                    print("🚨 FATAL: Recovery timed out. Shutting down motors!")
                                forSpd = 0.0
                                turnSpd = 0.0

                        # 2. Junction FSM Trigger
                        elif is_junction and lineFollow and junc_cooldown_count == 0:
                            if is_recovering:
                                is_recovering = False  # Cancel recovery if junction found
                            print(f"[JUNC] Detected '{current_label}' — entering STOPPING")
                            junc_state = JUNC_STOPPING
                            junc_phase_start = now
                            forSpd, turnSpd = 0.0, 0.0

                        # 3. Normal PID Tracking
                        else:
                            if is_recovering:
                                print("✅ SUCCESS: Line re-acquired! Resuming normal operation.")
                                is_recovering = False

                            kP = current_params["kP"]
                            kD = current_params["kD"]
                            col_bias = current_params["col_bias"]
                            spd_fac = current_params["spd"]

                            col_adj = (col + col_bias) if col is not None else col
                            forSpd, turnSpd = line2SpdMap.send((col_adj, kP, kD))
                            forSpd *= spd_fac

                            # Dynamic speed damping
                            turn_norm = min(abs(turnSpd) / TURN_SPD_MAX, 1.0)
                            speed_factor = 1.0 - turn_norm * DAMPING_MAX
                            forSpd *= speed_factor

                            last_valid_turn = turnSpd

                    elif junc_state == JUNC_STOPPING:
                        if now - junc_phase_start >= JUNC_STOP_DURATION:

                            if current_label == "crossroad":
                                current_action = turn_sequence[turn_index]
                                turn_index = (turn_index + 1) % len(turn_sequence)

                                if current_action == 0:
                                    junc_state = JUNC_DRIVING
                                    junc_phase_start = now
                                    print("[JUNC] Crossroad. Action: STRAIGHT through intersection.")
                                else:
                                    junc_turn_sign = current_action
                                    junc_state = JUNC_TURNING
                                    junc_phase_start = now
                                    print(f"[JUNC] Crossroad. Action: Turning "
                                          f"{'LEFT' if junc_turn_sign == 1 else 'RIGHT'} "
                                          f"for {JUNC_TURN_DURATION:.2f} s")

                            elif current_label == "t_junction":
                                junc_turn_sign = 1  # 1 = Left
                                junc_state = JUNC_TURNING
                                junc_phase_start = now
                                print(f"[JUNC] T-Junction. Action: Turning LEFT (Fixed) "
                                      f"for {JUNC_TURN_DURATION:.2f} s")

                    elif junc_state == JUNC_TURNING:
                        if now - junc_phase_start >= JUNC_TURN_DURATION:
                            junc_state = JUNC_DRIVING
                            junc_phase_start = now
                            print("[JUNC] Turn complete — entering DRIVING_THROUGH")

                    elif junc_state == JUNC_DRIVING:
                        time_in_phase = now - junc_phase_start
                        road_clear = current_label in ("straight", "curve")
                        if time_in_phase >= JUNC_MIN_DRIVE_TIME and road_clear:
                            print(f"[JUNC] Road clear (CNN='{current_label}') — resuming normal tracking")
                            junc_state = JUNC_FOLLOW
                            junc_cooldown_count = JUNC_COOLDOWN_FRAMES
                            current_params = SCENE_PARAMS.get(current_label, _DEFAULT)

                if counterDown % 4 == 0:
                    probe.send(name='Raw Image', imageData=gray_sm)
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