# Hand Gesture Recognition System

A real-time hand gesture recognition system built with **Python, OpenCV, MediaPipe, and Machine Learning**. The application detects hand gestures through a webcam and performs predefined desktop actions automatically.

The project uses **MediaPipe Hand Landmarks** for hand tracking and a **Random Forest Classifier** for gesture classification, achieving high accuracy while maintaining real-time performance.

---

## Features

* Real-time hand gesture recognition
* Webcam-based input (Laptop Camera or DroidCam)
* MediaPipe hand landmark extraction
* Machine Learning gesture classification
* Desktop automation based on recognized gestures
* Fast and lightweight Random Forest model
* High recognition accuracy

---

## Supported Gestures

| Gesture  | Action                |
| -------- | --------------------- |
| ✋ Palm   | Open YouTube          |
| ✊ Fist   | Show Desktop          |
| 👍 Thumb | Lock Computer         |
| ☝️ Index | Open Windows Settings |
| ✌️ Peace | Open File Explorer    |

---

## Technology Stack

* Python
* OpenCV
* MediaPipe
* Scikit-Learn
* NumPy
* Joblib

---

## Project Workflow

1. Capture hand landmarks using MediaPipe.
2. Extract landmark coordinates from each frame.
3. Generate a structured dataset.
4. Train a Random Forest classifier.
5. Predict gestures in real time.
6. Execute mapped desktop actions.

---

## Project Structure

```text
Hand Gesture Recognition/
│
├── dataset/
│   └── Final gesture landmark dataset
│
├── models/
│   ├── gesture_random_forest.joblib
│   └── hand_landmarker.task
│
├── results/
│   └── Evaluation results and predictions
│
├── splits/
│   └── Training and testing datasets
│
├── src/
│   ├── collect_landmarks.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── realtime_recognition.py
│
├── requirements.txt
├── README.md
└── Start Hand Gesture Recognition.bat
```

---

## Model Performance

| Metric            | Value                    |
| ----------------- | ------------------------ |
| Model             | Random Forest Classifier |
| Test Accuracy     | 98.58%                   |
| Number of Classes | 5                        |
| Dataset Size      | 6,000 Samples            |

The trained Random Forest model provides excellent accuracy while remaining lightweight enough for real-time deployment.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/parthtech221/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the gesture recognition system:

```bash
python src/realtime_recognition.py
```

Or use:

```bash
Start Hand Gesture Recognition.bat
```

---

## Camera Configuration

The application supports:

* Laptop Webcam
* USB Camera
* DroidCam

To change the camera source, modify:

```python
CAMERA_INDEX = 0
```

inside:

```text
src/realtime_recognition.py
```

Common values:

```text
0 = Laptop Webcam
1 = DroidCam / External Camera
2+ = Additional Cameras
```

---

## Future Improvements

* Additional gesture classes
* Custom user-defined actions
* Deep Learning-based gesture recognition
* Multi-hand support
* Cross-platform automation support
* Gesture-based volume and media controls

---

## Author

**Parth Patel**

AI / Machine Learning Enthusiast

GitHub: https://github.com/parthtech221

---

## License

This project is intended for educational and research purposes.
