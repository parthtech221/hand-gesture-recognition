import cv2
import csv
import time
import uuid
import math
import mediapipe as mp

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Samsung Galaxy S25 Ultra through DroidCam USB,& Camera = 0 for laptop camera directly
CAMERA_INDEX = 0

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "custom_landmarks"
OUTPUT_FILE = OUTPUT_DIR / "hand_landmarks.csv"

TARGET_SAMPLES = 150

# Time between possible samples
CAPTURE_INTERVAL = 0.12

# Movement/shape thresholds used to avoid near-duplicate samples
MOVEMENT_THRESHOLD = 0.006
SHAPE_THRESHOLD = 0.012

# Countdown before collection
COUNTDOWN_SECONDS = 3


# ============================================================
# FINAL FIVE GESTURES
# ============================================================

GESTURES = ["01_palm", "02_fist", "03_thumb", "04_index", "05_peace"]

PEOPLE = [
    "person_1",
    "person_2"
]

LIGHTING_OPTIONS = [
    "natural",
    "frontlight",
    "backlight",
    "sidelight"
]


# ============================================================
# MENU FUNCTION
# ============================================================

def choose_option(title, options):

    print(f"\n{title}")

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:

        choice = input("\nEnter option number: ").strip()

        if choice.isdigit():

            index = int(choice) - 1

            if 0 <= index < len(options):
                return options[index]

        print("Invalid choice. Please try again.")


# ============================================================
# LANDMARK CONNECTIONS
# ============================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    for start, end in CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )


# ============================================================
# NORMALIZE HAND SHAPE
# ============================================================

def normalized_shape(landmarks):

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    palm_size = math.sqrt(
        (middle_mcp.x - wrist.x) ** 2 +
        (middle_mcp.y - wrist.y) ** 2
    )

    palm_size = max(palm_size, 0.0001)

    normalized = []

    for landmark in landmarks:

        normalized.append(
            (
                (landmark.x - wrist.x) / palm_size,
                (landmark.y - wrist.y) / palm_size
            )
        )

    return normalized


# ============================================================
# CHECK WHETHER SAMPLE IS DIFFERENT ENOUGH
# ============================================================

def is_different_enough(current, previous):

    if previous is None:
        return True

    # --------------------------------------------------------
    # Hand position movement
    # --------------------------------------------------------

    movement = 0.0

    for a, b in zip(current, previous):

        movement += math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        )

    movement /= 21

    # --------------------------------------------------------
    # Hand shape movement
    # --------------------------------------------------------

    current_shape = normalized_shape(current)
    previous_shape = normalized_shape(previous)

    shape_difference = 0.0

    for a, b in zip(current_shape, previous_shape):

        shape_difference += math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    shape_difference /= 21

    return (
        movement >= MOVEMENT_THRESHOLD
        or shape_difference >= SHAPE_THRESHOLD
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\n==========================================")
print("       HAND GESTURE DATA COLLECTOR")
print("==========================================")

person = choose_option(
    "SELECT PERSON",
    PEOPLE
)

lighting = choose_option(
    "SELECT LIGHTING",
    LIGHTING_OPTIONS
)

gesture = choose_option(
    "SELECT GESTURE",
    GESTURES
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CSV HEADER
# ============================================================

HEADER = [
    "sample_id",
    "session_id",
    "timestamp",
    "person",
    "lighting",
    "gesture",
    "handedness"
]

for i in range(21):

    HEADER.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# ============================================================
# CHECK EXISTING SAMPLES
# ============================================================

existing_count = 0

if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != HEADER:

            raise ValueError(
                "\nExisting CSV has an unexpected format.\n"
                "Please check the file before collecting more data."
            )

        for row in reader:

            if (
                row["person"] == person
                and row["lighting"] == lighting
                and row["gesture"] == gesture
            ):

                existing_count += 1


if existing_count >= TARGET_SAMPLES:

    print("\nThis combination already has 150 samples.")
    print(f"Person   : {person}")
    print(f"Lighting : {lighting}")
    print(f"Gesture  : {gesture}")

    raise SystemExit


remaining = TARGET_SAMPLES - existing_count


# ============================================================
# SESSION INFORMATION
# ============================================================

session_id = uuid.uuid4().hex[:12]

print("\n==========================================")
print("             SESSION DETAILS")
print("==========================================")

print(f"Person          : {person}")
print(f"Lighting        : {lighting}")
print(f"Gesture          : {gesture}")
print(f"Existing samples : {existing_count}")
print(f"Remaining        : {remaining}")
print(f"Session ID       : {session_id}")

print("==========================================")

input("\nPress ENTER to open the camera...")


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"MediaPipe model not found:\n{MODEL_PATH}"
    )


base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    detector.close()

    raise RuntimeError(
        "Could not open S25 Ultra camera."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# OPEN CSV
# ============================================================

file_needs_header = (
    not OUTPUT_FILE.exists()
    or OUTPUT_FILE.stat().st_size == 0
)

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

if file_needs_header:

    writer.writerow(HEADER)
    csv_file.flush()


# ============================================================
# COLLECTION STATE
# ============================================================

sample_count = existing_count

previous_landmarks = None

last_capture_time = 0

state = "WAITING"

countdown_start = None


print("\nCamera ready.")
print("SPACE = Start")
print("P     = Pause")
print("Q     = Quit")


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:

            print("Could not read camera frame.")
            break


        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect(mp_image)


        # ----------------------------------------------------
        # Detect one hand
        # ----------------------------------------------------

        hand_detected = len(result.hand_landmarks) == 1

        landmarks = None

        handedness = "Unknown"

        if hand_detected:

            landmarks = result.hand_landmarks[0]

            draw_hand(
                frame,
                landmarks
            )

            if (
                result.handedness
                and result.handedness[0]
            ):

                handedness = (
                    result.handedness[0][0].category_name
                )


        now = time.monotonic()

        status = ""


        # ====================================================
        # COUNTDOWN
        # ====================================================

        if state == "COUNTDOWN":

            elapsed = now - countdown_start

            remaining_time = max(
                0,
                math.ceil(
                    COUNTDOWN_SECONDS - elapsed
                )
            )

            if remaining_time > 0:

                status = (
                    f"Starting in {remaining_time}..."
                )

            else:

                state = "COLLECTING"

                previous_landmarks = None
                last_capture_time = 0

                status = "COLLECTING..."


        # ====================================================
        # COLLECTION
        # ====================================================

        if state == "COLLECTING":

            if not hand_detected:

                status = "No hand detected"

            else:

                enough_time = (
                    now - last_capture_time
                    >= CAPTURE_INTERVAL
                )

                different_enough = (
                    is_different_enough(
                        landmarks,
                        previous_landmarks
                    )
                )

                if enough_time and different_enough:

                    sample_id = uuid.uuid4().hex

                    timestamp = (
                        datetime.now()
                        .astimezone()
                        .isoformat(
                            timespec="milliseconds"
                        )
                    )

                    row = [
                        sample_id,
                        session_id,
                        timestamp,
                        person,
                        lighting,
                        gesture,
                        handedness
                    ]

                    # ----------------------------------------
                    # Store 21 × 3 landmark coordinates
                    # ----------------------------------------

                    for landmark in landmarks:

                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    writer.writerow(row)
                    csv_file.flush()

                    sample_count += 1

                    # Save current landmarks as reference
                    previous_landmarks = [
                        SimpleNamespace(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z
                        )
                        for landmark in landmarks
                    ]

                    last_capture_time = now

                    status = "Sample saved"

                else:

                    status = "Collecting..."


                # --------------------------------------------
                # Completion
                # --------------------------------------------

                if sample_count >= TARGET_SAMPLES:

                    state = "COMPLETE"


        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (650, 245),
            (20, 20, 20),
            -1
        )

        display_lines = [
            f"Person: {person}",
            f"Lighting: {lighting}",
            f"Gesture: {gesture}",
            f"Hand: {handedness}",
            f"Samples: {sample_count}/{TARGET_SAMPLES}",
            f"State: {state}",
            status
        ]

        for i, line in enumerate(display_lines):

            color = (255, 255, 255)

            if i == 4:
                color = (0, 255, 255)

            if i == 6:
                color = (0, 255, 0)

            cv2.putText(
                frame,
                line,
                (20, 40 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )


        cv2.putText(
            frame,
            "SPACE: Start | P: Pause | Q: Quit",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Hand Gesture Dataset Collector",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\nCollection stopped.")
            break


        if key == ord("p"):

            if state == "COLLECTING":

                state = "PAUSED"

                print("\nCollection paused.")


            elif state == "PAUSED":

                state = "COUNTDOWN"

                countdown_start = time.monotonic()

                print("\nResuming...")


        if key == ord(" ") and state in (
            "WAITING",
            "PAUSED"
        ):

            state = "COUNTDOWN"

            countdown_start = time.monotonic()

            print("\nCountdown started...")


        if state == "COMPLETE":

            print("\nTarget reached.")
            break


finally:

    csv_file.close()

    cap.release()

    cv2.destroyAllWindows()

    detector.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("          COLLECTION COMPLETE")
print("==========================================")

print(f"Person          : {person}")
print(f"Lighting        : {lighting}")
print(f"Gesture         : {gesture}")
print(f"New samples     : {sample_count - existing_count}")
print(f"Total samples   : {sample_count}/{TARGET_SAMPLES}")
print(f"Saved to        : {OUTPUT_FILE}")

print("==========================================")