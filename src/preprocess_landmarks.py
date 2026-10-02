from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
FEATURES_DIR = ROOT / "features"

INPUT_FILE = DATASET_DIR / "hand_gesture_master_dataset.csv"
OUTPUT_FILE = FEATURES_DIR / "processed_landmarks.csv"
LABEL_MAP_FILE = FEATURES_DIR / "label_map.csv"


# ============================================================
# GESTURE LABELS
# ============================================================

GESTURES = ["01_palm", "02_fist", "03_thumb", "04_index", "05_peace"]

LABEL_MAP = {gesture: i for i, gesture in enumerate(GESTURES)}


# ============================================================
# LANDMARK COLUMNS
# ============================================================

# MediaPipe provides 21 hand landmarks.
# Each landmark has x, y, z coordinates.
# Total = 21 × 3 = 63 features.

LANDMARK_COLS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_landmarks(df):
    """
    Convert raw MediaPipe hand landmarks into normalized
    63-dimensional feature vectors.

    Steps:
    1. Wrist centering
    2. Hand-size normalization
    3. Handedness normalization
    """

    # --------------------------------------------------------
    # Convert dataframe landmarks to NumPy array
    # Shape:
    #     (samples, 21, 3)
    # --------------------------------------------------------

    xyz = (
        df[LANDMARK_COLS]
        .to_numpy(dtype=np.float64)
        .reshape(-1, 21, 3)
    )

    # --------------------------------------------------------
    # STEP 1: Wrist centering
    #
    # Landmark 0 = wrist.
    #
    # After this operation:
    #     wrist = (0, 0, 0)
    #
    # This removes the effect of where the hand appears
    # inside the camera frame.
    # --------------------------------------------------------

    xyz = xyz - xyz[:, 0:1, :]

    # --------------------------------------------------------
    # STEP 2: Scale normalization
    #
    # Landmark 9 = middle finger MCP joint.
    #
    # We use the distance:
    #     wrist → middle MCP
    #
    # as the reference hand size.
    #
    # This reduces differences caused by:
    # - hand size
    # - distance from camera
    # --------------------------------------------------------

    scale = np.linalg.norm(
        xyz[:, 9, :],
        axis=1
    )

    # Prevent division by zero.
    scale[scale < 1e-8] = 1.0

    xyz = xyz / scale[:, None, None]

    # --------------------------------------------------------
    # STEP 3: Handedness normalization
    #
    # Left hands are mirrored along the X-axis so that
    # left and right hands have the same canonical orientation.
    # --------------------------------------------------------

    left_hand = (
        df["handedness"]
        .astype(str)
        .str.lower()
        .eq("left")
        .to_numpy()
    )

    xyz[left_hand, :, 0] *= -1.0

    # --------------------------------------------------------
    # Convert back to 63 features
    #
    # Shape:
    #     (samples, 63)
    # --------------------------------------------------------

    return xyz.reshape(-1, 63)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("HAND GESTURE LANDMARK PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:")
        print(INPUT_FILE)
        print("\nMake sure the master CSV is inside the dataset folder.")
        return

    # Create features directory if required.
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Raw samples: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "sample_id",
        "session_id",
        "timestamp",
        "person",
        "lighting",
        "gesture",
        "handedness",
    ] + LANDMARK_COLS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing columns:")
        for col in missing_columns:
            print(f"  - {col}")
        return

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = df[required_columns].isnull().sum().sum()

    print(f"Missing values: {missing_values}")

    if missing_values > 0:
        print("\nERROR: Dataset contains missing values.")
        return

    # --------------------------------------------------------
    # Preprocess landmarks
    # --------------------------------------------------------

    print("\nApplying preprocessing:")
    print("  1. Wrist centering")
    print("  2. Scale normalization")
    print("  3. Handedness normalization")

    X = preprocess_landmarks(df)

    # --------------------------------------------------------
    # Create processed dataframe
    # --------------------------------------------------------

    processed_columns = [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

    processed_df = pd.DataFrame(
        X,
        columns=processed_columns
    )

    # Add useful metadata.
    processed_df.insert(
        0,
        "sample_id",
        df["sample_id"].values
    )

    processed_df.insert(
        1,
        "session_id",
        df["session_id"].values
    )

    processed_df.insert(
        2,
        "person",
        df["person"].values
    )

    processed_df.insert(
        3,
        "lighting",
        df["lighting"].values
    )

    processed_df.insert(
        4,
        "gesture",
        df["gesture"].values
    )

    processed_df.insert(
        5,
        "label",
        df["gesture"].map(LABEL_MAP).values
    )

    # --------------------------------------------------------
    # Validate processed features
    # --------------------------------------------------------

    feature_values = processed_df[processed_columns].to_numpy()

    if np.isnan(feature_values).any():
        print("\nERROR: NaN values detected after preprocessing.")
        return

    if not np.isfinite(feature_values).all():
        print("\nERROR: Infinite values detected after preprocessing.")
        return

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save label map
    # --------------------------------------------------------

    label_map_df = pd.DataFrame(
        {
            "label": list(LABEL_MAP.values()),
            "gesture": list(LABEL_MAP.keys()),
        }
    )

    label_map_df.to_csv(
        LABEL_MAP_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nSamples processed : {len(processed_df)}")
    print(f"Landmark features : 63")
    print(f"Total output cols : {len(processed_df.columns)}")

    print("\nGesture labels:")

    for gesture, label in LABEL_MAP.items():
        count = (df["gesture"] == gesture).sum()
        print(f"  {label} -> {gesture}: {count} samples")

    print("\nOutput files:")

    print(f"  {OUTPUT_FILE}")
    print(f"  {LABEL_MAP_FILE}")

    print("\nValidation:")
    print("  NaN values       : 0")
    print("  Infinite values  : 0")

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()