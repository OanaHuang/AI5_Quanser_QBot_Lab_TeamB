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
#  Scene Parameter Table  (crossroad / t_junction params are now only used
#  for normal PID fallback; the junction state machine takes over instead)
# ─────────────────────────────────────────────
SCENE_PARAMS = {
    # label        kP     kD    col_bias  spd_factor
    "straight":  {"kP": 0.4, "kD": 0.4, "col_bias":   0, "spd": 0.40},
    "curve":     {"kP": 0.4, "kD": 0.4, "col_bias":   0, "spd": 0.40},
    "crossroad": {"kP": 2.5, "kD": 0.6, "col_bias": -200, "spd": -0.05},
    "t_junction":{"kP": 2.5, "kD": 0.6, "col_bias": -200, "spd": -0.05},
    "out_route": {"kP": 0.4, "kD": 0.4, "col_bias":   0, "spd": 0.4},
}
_DEFAULT = {"kP": 0.40, "kD": 0.40, "col_bias": 0, "spd": 0.3}
CNN_CONF_THRESHOLD = 0.50   # Fall back to default values when confidence is below this threshold
CNN_INFER_EVERY    = 30     # Run inference every N frames (30 frames ≈ 0.5s @ 60Hz)

# ─────────────────────────────────────────────
#  Junction State Machine — Tunable Parameters
#
#  JUNC_STOP_DURATION  : seconds the robot stays fully stopped before turning.
#  JUNC_TURN_SPEED     : angular velocity (rad/s) used for the 90° turn.
#                        Positive value → right, negative → left (applied by the
#                        state machine with the correct sign after image scan).
#  JUNC_TURN_DURATION  : time (s) to sustain the turn = (π/2) / JUNC_TURN_SPEED.
#                        Adjust if the physical platform overshoots or undershoots.
#  JUNC_DRIVE_SPEED    : forward speed (m/s) while driving through the intersection
#                        after the turn, before CNN re-acquires a straight/curve.
#  JUNC_MIN_DRIVE_TIME : minimum seconds to drive forward after the turn before
#                        allowing the CNN to end the DRIVING_THROUGH phase.
#                        Prevents premature exit while still over the junction markings.
#  JUNC_COOLDOWN_FRAMES: camera frames to ignore junction detections after the full
#                        maneuver completes, preventing immediate re-trigger.
# ─────────────────────────────────────────────
JUNC_STOP_DURATION   = 0.5
JUNC_TURN_SPEED      = 0.50                          # rad/s — tune to your platform
JUNC_TURN_DURATION   = (np.pi / 2) / JUNC_TURN_SPEED  # ≈ 3.14 s for 90°
JUNC_DRIVE_SPEED     = 0.15                          # m/s while exiting intersection
JUNC_MIN_DRIVE_TIME  = 0.5                           # s before CNN can end DRIVING phase
JUNC_COOLDOWN_FRAMES = 90                            # frames (~1.5 s @ 60 Hz)

# ─────────────────────────────────────────────
#  Dynamic Speed Damping (Solution A)
#
#  When the robot is steering hard, forward speed is reduced automatically
#  so the PID has time to correct before the robot drifts further off-line.
#
#  TURN_SPD_MAX   : the maximum |turnSpd| your platform can output
#                   (matches the `saturation` argument in line_to_speed_map).
#                   Used to normalise the turn magnitude to [0, 1].
#  DAMPING_MAX    : maximum fraction of forward speed that can be removed.
#                   0.6 → at full steering the robot keeps 40 % of forSpd.
#                   Raise toward 1.0 for tighter tracks; lower for smoother rides.
# ─────────────────────────────────────────────
TURN_SPD_MAX  = 75.0   # matches saturation=75 in line_to_speed_map
DAMPING_MAX   = 0.60   # clamp: never remove more than 60 % of forward speed

# Junction state identifiers
JUNC_FOLLOW         = "follow"
JUNC_STOPPING       = "stopping"
JUNC_TURNING        = "turning"
JUNC_DRIVING        = "driving_through"

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


def scan_turn_direction(gray_image: np.ndarray,
                        row_start: int = 50, row_end: int = 100,
                        min_thresh: int = 100, max_thresh: int = 255) -> int:
    """
    Examine the white-pixel distribution in a horizontal strip of the grayscale
    image and return the preferred turn direction.

    Strategy
    --------
    The image is split vertically at the centre column (width // 2).
    White pixels (brightness between min_thresh and max_thresh) are counted in
    the left half and the right half independently.  The robot should turn toward
    the side that has MORE white pixels — that is where the continuation of the
    road lies.

    Returns
    -------
    +1  →  turn RIGHT  (more white on the right)
    -1  →  turn LEFT   (more white on the left, or a tie)
    """
    strip  = gray_image[row_start:row_end, :]
    mask   = (strip > min_thresh) & (strip < max_thresh)
    mid    = strip.shape[1] // 2
    left_count  = int(mask[:, :mid].sum())
    right_count = int(mask[:, mid:].sum())

    direction = 1 if right_count > left_count else -1
    print(f"[JUNC] Scan → left_white={left_count}  right_white={right_count}"
          f"  → turn {'RIGHT' if direction == 1 else 'LEFT'}")
    return direction


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

# ── Junction state machine runtime variables ──────────────────────────────────
junc_state          = JUNC_FOLLOW   # current state
junc_phase_start    = 0.0           # wall-clock time when the current phase began
junc_turn_sign      = 1             # +1 = right, -1 = left (set during STOPPING→TURNING)
junc_cooldown_count = 0             # frames remaining in post-maneuver cooldown
# ─────────────────────────────────────────────────────────────────────────────

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

    # Ensure binary is always defined before the probe.send call
    binary = np.zeros((50, 320), dtype=np.uint8)

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

            # ─────────────────────────────────────────────────────────────────
            #  Section C - Command Selection
            #
            #  Priority order:
            #    1. Manual keyboard override (lineFollow=False)
            #    2. Junction state machine  (junc_state != JUNC_FOLLOW)
            #    3. Normal PID line following
            # ─────────────────────────────────────────────────────────────────
            if not lineFollow:
                # Manual keyboard control — state machine is paused
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)

            elif junc_state == JUNC_STOPPING:
                # ── Phase 1: Hold still ───────────────────────────────────────
                commands = np.array([0.0, 0.0], dtype=np.float64)

            elif junc_state == JUNC_TURNING:
                # ── Phase 2: Spin in place toward the detected road side ──────
                commands = np.array([0.0, junc_turn_sign * JUNC_TURN_SPEED],
                                    dtype=np.float64)

            elif junc_state == JUNC_DRIVING:
                # ── Phase 3: Drive straight through the intersection ──────────
                commands = np.array([JUNC_DRIVE_SPEED, 0.0], dtype=np.float64)

            else:
                # ── Normal PID line following (JUNC_FOLLOW) ───────────────────
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
                    #  Section D.2 - Threshold & Blob
                    #  (always computed — needed by both PID and junction scan)
                    # ─────────────────────────────────────
                    rowStart, rowEnd           = 50, 100
                    minThreshold, maxThreshold = 100, 255

                    subImage = gray_sm[rowStart:rowEnd, :]
                    binary   = np.zeros_like(subImage)
                    h_sub, w_sub = subImage.shape

                    for i in range(h_sub):
                        for j in range(w_sub):
                            if minThreshold < subImage[i, j] < maxThreshold:
                                binary[i, j] = 255

                    connectivity = 8
                    min_pixels, max_pixels = 500, 2000
                    col, row, area = vision.image_find_objects(
                        binary, connectivity, min_pixels, max_pixels
                    )

                    # ─────────────────────────────────────
                    #  Section D.CNN - Scene Classification
                    #  Suppressed while inside the junction maneuver to prevent
                    #  the robot from re-classifying mid-turn.
                    # ─────────────────────────────────────
                    if (junc_state == JUNC_FOLLOW and
                            counterDown % CNN_INFER_EVERY == 1):
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

                    # Also run CNN during DRIVING_THROUGH (after min drive time)
                    # so we can detect when the road is clear of junction markings.
                    if (junc_state == JUNC_DRIVING and
                            counterDown % CNN_INFER_EVERY == 1):
                        new_label, new_conf, _ = infer_scene(
                            cnn_model, gray_sm, CNN_CONF_THRESHOLD
                        )
                        current_label = new_label
                        current_conf  = new_conf

                    # ──────────────────────────────────────────────────────────
                    #  Section D.JUNC - Junction State Machine Transitions
                    # ──────────────────────────────────────────────────────────

                    # Decrement cooldown counter
                    if junc_cooldown_count > 0:
                        junc_cooldown_count -= 1

                    now = elapsed_time()

                    if junc_state == JUNC_FOLLOW:
                        # ── Trigger: CNN sees a junction while not in cooldown ──
                        is_junction = current_label in ("crossroad", "t_junction")
                        if is_junction and lineFollow and junc_cooldown_count == 0:
                            print(f"[JUNC] Detected '{current_label}' — entering STOPPING")
                            junc_state       = JUNC_STOPPING
                            junc_phase_start = now
                            forSpd, turnSpd  = 0.0, 0.0   # freeze PID output

                    elif junc_state == JUNC_STOPPING:
                        # ── Wait until fully stopped, then scan & transition ───
                        if now - junc_phase_start >= JUNC_STOP_DURATION:
                            # Scan the current binary image to decide turn direction
                            junc_turn_sign   = scan_turn_direction(gray_sm, rowStart, rowEnd,
                                                                   minThreshold, maxThreshold)
                            junc_state       = JUNC_TURNING
                            junc_phase_start = now
                            print(f"[JUNC] Stopped. Turning "
                                  f"{'RIGHT' if junc_turn_sign == 1 else 'LEFT'} "
                                  f"for {JUNC_TURN_DURATION:.2f} s")

                    elif junc_state == JUNC_TURNING:
                        # ── Spin for the pre-calculated 90° duration ──────────
                        if now - junc_phase_start >= JUNC_TURN_DURATION:
                            junc_state       = JUNC_DRIVING
                            junc_phase_start = now
                            print("[JUNC] Turn complete — entering DRIVING_THROUGH")

                    elif junc_state == JUNC_DRIVING:
                        # ── Drive straight until CNN re-acquires straight/curve ─
                        time_in_phase = now - junc_phase_start
                        road_clear    = current_label in ("straight", "curve")
                        if time_in_phase >= JUNC_MIN_DRIVE_TIME and road_clear:
                            print(f"[JUNC] Road clear (CNN='{current_label}') — "
                                  f"resuming normal line following")
                            junc_state          = JUNC_FOLLOW
                            junc_cooldown_count = JUNC_COOLDOWN_FRAMES
                            # Restore PID params for the detected road type
                            current_params = SCENE_PARAMS.get(current_label, _DEFAULT)

                    # ──────────────────────────────────────────────────────────
                    #  Section D.3 - Adaptive Speed Command  (PID line following)
                    #  Only updates forSpd / turnSpd when in normal FOLLOW state.
                    # ──────────────────────────────────────────────────────────
                    if junc_state == JUNC_FOLLOW:
                        kP       = current_params["kP"]
                        kD       = current_params["kD"]
                        col_bias = current_params["col_bias"]
                        spd_fac  = current_params["spd"]

                        col_adj = (col + col_bias) if col is not None else col
                        forSpd, turnSpd = line2SpdMap.send((col_adj, kP, kD))
                        forSpd *= spd_fac

                        # ── Solution A: dynamic speed damping ─────────────────
                        # Larger steering demand → more forward speed removed.
                        # At turnSpd=0 the factor is 1.0 (no effect).
                        # At |turnSpd|=TURN_SPD_MAX the factor drops to (1-DAMPING_MAX).
                        turn_norm    = min(abs(turnSpd) / TURN_SPD_MAX, 1.0)
                        speed_factor = 1.0 - turn_norm * DAMPING_MAX
                        forSpd      *= speed_factor

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