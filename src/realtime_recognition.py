from pathlib import Path
from collections import deque
import ctypes
import subprocess
import time
from ctypes import wintypes

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================== PATHS ==============================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "gesture_random_forest.joblib"
LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"


# ============================= SETTINGS ============================

# Camera 1 = Samsung Galaxy S25 Ultra through DroidCam. & Camera = 0 for laptop camera directly
CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360

SMOOTHING_WINDOW = 8
CONFIDENCE_THRESHOLD = 0.85
GESTURE_HOLD_TIME = 2.0
GRACE_PERIOD = 5.0

# Briefly show "0" before executing the action.
ACTION_ZERO_DISPLAY = 0.20

# Let the external application remain visible before returning focus
# to the recognition window.
ACTION_RETURN_DELAY = 2.0

CAMERA_WIDTH = 560
CAMERA_HEIGHT = 360
PANEL_HEIGHT = 200
WINDOW_WIDTH = CAMERA_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT + PANEL_HEIGHT
WINDOW_TITLE = "Hand Gesture Recognition"

PROBABILITY_MARGIN_THRESHOLD = 0.10
ACTION_DISPLAY_TIME = 0.35


# ============================== LABELS =============================

LABEL_MAP = {
    0: "Palm",
    1: "Fist",
    2: "Thumb",
    3: "Index",
    4: "Peace",
}


# ========================= WINDOWS HELPERS =========================

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

SW_RESTORE = 9
SW_MINIMIZE = 6
SW_SHOW = 5

HWND_TOP = 0
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040

EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL,
    wintypes.HWND,
    wintypes.LPARAM,
)


def get_window_title(hwnd):
    """Return the title of a top-level Windows window."""
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""

    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def find_visible_window(title_keywords):
    """Find a visible top-level window whose title contains a keyword."""
    keywords = [keyword.lower() for keyword in title_keywords]
    found_hwnd = None

    def callback(hwnd, _):
        nonlocal found_hwnd

        if not user32.IsWindowVisible(hwnd):
            return True

        title = get_window_title(hwnd)
        if not title:
            return True

        title_lower = title.lower()
        if any(keyword in title_lower for keyword in keywords):
            found_hwnd = hwnd
            return False

        return True

    callback_proc = EnumWindowsProc(callback)
    user32.EnumWindows(callback_proc, 0)
    return found_hwnd


def activate_window(hwnd):
    """Force a normal window into the foreground without leaving it topmost."""
    if not hwnd or not user32.IsWindow(hwnd):
        return False

    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    else:
        user32.ShowWindow(hwnd, SW_SHOW)

    current_thread = kernel32.GetCurrentThreadId()
    foreground = user32.GetForegroundWindow()
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None)
        if foreground else 0
    )
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)

    attached_foreground = False
    attached_target = False

    try:
        if foreground_thread and foreground_thread != current_thread:
            attached_foreground = bool(
                user32.AttachThreadInput(
                    current_thread, foreground_thread, True
                )
            )

        if target_thread and target_thread != current_thread:
            attached_target = bool(
                user32.AttachThreadInput(
                    current_thread, target_thread, True
                )
            )

        # Temporarily unlock foreground activation. This is especially
        # useful when Windows Settings (ApplicationFrameHost.exe) is
        # currently the foreground application.
        user32.LockSetForegroundWindow(2)  # LSFW_UNLOCK

        # A brief ALT key press is a standard Windows workaround for the
        # foreground-lock restriction on SetForegroundWindow.
        VK_MENU = 0x12
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

        # Temporarily raise the recognition window above normal windows.
        # Immediately remove TOPMOST so it does not stay permanently above
        # Chrome, Settings, File Explorer, etc.
        user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
        )
        user32.SetWindowPos(
            hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
        )

        user32.BringWindowToTop(hwnd)
        user32.SetActiveWindow(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)

        # SwitchToThisWindow is useful for modern Windows application
        # windows when SetForegroundWindow alone is ignored.
        if hasattr(user32, "SwitchToThisWindow"):
            user32.SwitchToThisWindow(hwnd, True)

        # Retry briefly because Settings can release foreground ownership
        # asynchronously when its navigation window is closing/opening.
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline:
            if user32.GetForegroundWindow() == hwnd:
                return True

            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)

        return user32.GetForegroundWindow() == hwnd

    finally:
        if attached_target:
            user32.AttachThreadInput(
                current_thread, target_thread, False
            )
        if attached_foreground:
            user32.AttachThreadInput(
                current_thread, foreground_thread, False
            )


def bring_window_to_front(title_keywords, timeout=2.0):
    """
    Find and activate a visible window by title keywords.

    This does NOT make the window permanently topmost.
    """
    end_time = time.monotonic() + timeout

    while time.monotonic() < end_time:
        hwnd = find_visible_window(title_keywords)

        if hwnd and activate_window(hwnd):
            return True

        time.sleep(0.05)

    return False


def bring_recognition_window_to_front():
    """Return focus to the OpenCV recognition window."""
    return bring_window_to_front([WINDOW_TITLE], timeout=1.5)


# ============================ ACTIONS ===============================

def open_chrome():
    print("ACTION: Opening YouTube...")
    subprocess.Popen(
        ["cmd", "/c", "start", "", "https://www.youtube.com"],
        shell=False,
    )


def open_file_manager():
    print("ACTION: Opening File Explorer...")
    subprocess.Popen(["explorer.exe"])

    # Give Explorer time to create its window, then explicitly foreground it.
    time.sleep(0.3)
    bring_window_to_front(["File Explorer"], timeout=2.0)


def open_settings():
    print("ACTION: Opening Windows Settings...")
    subprocess.Popen(
        ["cmd", "/c", "start", "", "ms-settings:"],
        shell=False,
    )

    # Settings may take a moment to create its ApplicationFrameWindow.
    time.sleep(0.5)
    bring_window_to_front(["Settings"], timeout=2.0)


def lock_computer():
    print("ACTION: Locking computer...")
    subprocess.run(
        ["rundll32.exe", "user32.dll,LockWorkStation"],
        check=False,
    )


def show_desktop_keep_recognition():
    """Minimize visible application windows except the recognition window."""
    recognition_title = WINDOW_TITLE

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True

        title = get_window_title(hwnd)
        if not title:
            return True

        if recognition_title.lower() in title.lower():
            return True

        user32.ShowWindow(hwnd, SW_MINIMIZE)
        return True

    callback_proc = EnumWindowsProc(callback)
    user32.EnumWindows(callback_proc, 0)
    print("ACTION: Show Desktop")


ACTION_MAP = {
    "Palm": open_chrome,
    "Fist": show_desktop_keep_recognition,
    "Thumb": lock_computer,
    "Index": open_settings,
    "Peace": open_file_manager,
}

ACTION_NAMES = {
    "Palm": "Open YouTube",
    "Fist": "Show Desktop",
    "Thumb": "Lock Computer",
    "Index": "Open Settings",
    "Peace": "Open File Explorer",
}


# ============================== MODEL ===============================

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")


# ============================ MEDIAPIPE =============================

print("Loading MediaPipe Hand Landmarker...")

base_options = python.BaseOptions(
    model_asset_path=str(LANDMARKER_PATH),
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)
print("MediaPipe Hand Landmarker loaded successfully.")


# ======================= LANDMARK PREPROCESSING =====================

def preprocess_landmarks(landmarks, handedness):
    """
    Apply the same preprocessing used during training:
    1. Wrist-centering
    2. Scale normalization by wrist -> middle MCP
    3. Mirror left hands
    4. Flatten to 63 features
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    points -= points[0]

    scale = np.linalg.norm(points[9])
    if scale < 1e-6:
        return None

    points /= scale

    if handedness:
        hand_label = handedness[0].category_name.lower()
        if hand_label == "left":
            points[:, 0] *= -1

    features = points.flatten()
    return features.reshape(1, -1)


# ========================== LANDMARK DRAWING ========================

def draw_hand_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for start, end in connections:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )


# ============================ UI PANEL ==============================

def draw_panel(
    gesture_name,
    confidence,
    status,
    action_text=None,
    countdown_text=None,
    grace_text=None,
):
    panel = np.full(
        (PANEL_HEIGHT, WINDOW_WIDTH, 3),
        35,
        dtype=np.uint8,
    )

    x = 20
    y = 28
    line_gap = 25

    gesture_display = gesture_name if gesture_name else "—"
    confidence_display = (
        f"{confidence * 100:.1f}%"
        if confidence is not None
        else "—"
    )

    cv2.putText(
        panel,
        f"Gesture       : {gesture_display}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"Confidence    : {confidence_display}",
        (x, y + line_gap),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"Status        : {status}",
        (x, y + 2 * line_gap),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if action_text:
        cv2.putText(
            panel,
            f"Action        : {action_text}",
            (x, y + 3 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if countdown_text:
        cv2.putText(
            panel,
            f"Action in     : {countdown_text}",
            (x, y + 3 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if grace_text:
        cv2.putText(
            panel,
            f"Grace Period  : {grace_text}",
            (x, y + 3 * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        "Threshold: 85% | Hold: 2 sec | Grace: 5 sec",
        (x, PANEL_HEIGHT - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "Press Q to Quit",
        (x, PANEL_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return panel


# ============================== CAMERA ==============================

print()
print("Opening camera...")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}."
    )

print("Camera opened successfully.")
print()
print("Controls:")
print("  Q = Quit")
print()


# ============================ APPLICATION ===========================

probability_history = deque(maxlen=SMOOTHING_WINDOW)
probability_sum = np.zeros(len(LABEL_MAP), dtype=np.float32)
last_timestamp_ms = 0

gesture_start_time = None
grace_start_time = None
action_zero_time = None
action_return_time = None
pending_action_gesture = None
current_action_text = None

cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)

screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

window_x = max(0, screen_width - WINDOW_WIDTH - 10)
window_y = max(0, screen_height - WINDOW_HEIGHT - 50)

cv2.moveWindow(
    WINDOW_TITLE,
    window_x,
    window_y,
)


# ============================= MAIN LOOP ============================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    now = time.monotonic()
    timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
    last_timestamp_ms = timestamp_ms

    gesture_name = None
    top_probability = None

    # Convert BGR -> RGB for MediaPipe.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = detector.detect_for_video(mp_image, timestamp_ms)

    # --------------------------- Recognition -------------------------

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        draw_hand_landmarks(frame, landmarks)

        features = preprocess_landmarks(
            landmarks,
            result.handedness[0],
        )

        if features is not None:
            probabilities = model.predict_proba(features)[0].astype(np.float32, copy=False)
            if len(probability_history) == probability_history.maxlen:
                probability_sum -= probability_history[0]
            probability_history.append(probabilities)
            probability_sum += probabilities
            smoothed_probabilities = probability_sum / len(probability_history)

            sorted_indices = np.argsort(
                smoothed_probabilities
            )[::-1]

            top_index = sorted_indices[0]
            second_index = sorted_indices[1]

            top_probability = float(
                smoothed_probabilities[top_index]
            )
            second_probability = float(
                smoothed_probabilities[second_index]
            )

            probability_margin = (
                top_probability - second_probability
            )

            if (
                top_probability >= CONFIDENCE_THRESHOLD
                and probability_margin >= PROBABILITY_MARGIN_THRESHOLD
            ):
                gesture_name = LABEL_MAP[top_index]
    else:
        probability_sum.fill(0)
        probability_history.clear()

    # --------------------------- UI state ----------------------------

    status = "Ready"
    action_text = None
    countdown_text = None
    grace_text = None

    # -------------------------- Grace period -------------------------

    if grace_start_time is not None:
        grace_elapsed = now - grace_start_time

        if grace_elapsed < GRACE_PERIOD:
            status = "Grace Period"

            grace_remaining = GRACE_PERIOD - grace_elapsed
            grace_text = str(int(np.ceil(grace_remaining)))

            gesture_start_time = None
        else:
            grace_start_time = None
            current_action_text = None
            pending_action_gesture = None
            gesture_start_time = None

    # --------------------- Return recognition window -----------------

    if (
        action_return_time is not None
        and now >= action_return_time
    ):
        bring_recognition_window_to_front()
        action_return_time = None

    # ---------------------- Gesture hold timing ----------------------

    if (
        grace_start_time is None
        and action_zero_time is None
        and gesture_name in ACTION_MAP
        and top_probability is not None
        and top_probability >= CONFIDENCE_THRESHOLD
    ):
        if pending_action_gesture != gesture_name:
            pending_action_gesture = gesture_name
            gesture_start_time = now

        if gesture_start_time is not None:
            hold_elapsed = now - gesture_start_time

            if hold_elapsed < GESTURE_HOLD_TIME:
                status = "Hold"

                hold_remaining = (
                    GESTURE_HOLD_TIME - hold_elapsed
                )
                countdown_text = str(
                    int(np.ceil(hold_remaining))
                )
            else:
                status = "Hold"
                countdown_text = "0"

                if action_zero_time is None:
                    action_zero_time = now

    elif (
        grace_start_time is None
        and action_zero_time is None
        and gesture_name is None
    ):
        gesture_start_time = None
        pending_action_gesture = None

    # ------------------------- Execute action -----------------------

    if action_zero_time is not None:
        if now - action_zero_time >= ACTION_ZERO_DISPLAY:
            gesture_to_execute = pending_action_gesture

            if gesture_to_execute in ACTION_MAP:
                current_action_text = ACTION_NAMES[
                    gesture_to_execute
                ]

                print(
                    "Stable gesture detected: "
                    f"{gesture_to_execute}"
                )
                print(
                    "ACTION: "
                    f"{current_action_text}"
                )

                ACTION_MAP[gesture_to_execute]()

                # Return the recognition window after the external
                # application has had time to appear.
                action_return_time = (
                    time.monotonic()
                    + ACTION_RETURN_DELAY
                )

                # Start grace period immediately after the action.
                grace_start_time = time.monotonic()

            action_zero_time = None
            gesture_start_time = None
            pending_action_gesture = None

    # -------------------------- Action display ----------------------

    if (
        grace_start_time is not None
        and current_action_text is not None
        and time.monotonic() - grace_start_time < ACTION_DISPLAY_TIME
    ):
        status = "Action"
        action_text = current_action_text
        grace_text = None

    # ---------------------------- Display ----------------------------

    camera_view = cv2.resize(
        frame,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    panel = draw_panel(
        gesture_name,
        top_probability,
        status,
        action_text=action_text,
        countdown_text=countdown_text,
        grace_text=grace_text,
    )

    combined_view = np.vstack(
        (camera_view, panel)
    )

    cv2.imshow(
        WINDOW_TITLE,
        combined_view,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================== CLEANUP =============================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Program closed.")
