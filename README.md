# Hand Gesture Recognition

A real-time computer vision system that recognizes five predefined hand gestures from a camera feed and performs a mapped desktop action.

## Final gestures and actions

| Gesture | Action |
|---|---|
| Palm | Open YouTube |
| Fist | Show Desktop |
| Thumb | Lock Computer |
| Index | Open Windows Settings |
| Peace | Open File Explorer |

## Project structure

- `dataset/` — final 5-class landmark dataset (6,000 samples)
- `models/` — deployed Random Forest and MediaPipe Hand Landmarker model
- `splits/` — session-level train/test split used for evaluation
- `results/` — final evaluation metrics and test predictions
- `src/` — data collection, preprocessing, splitting, training, and real-time recognition scripts
- `requirements.txt` — project dependencies

## Model result

Random Forest test accuracy: **98.58%**

The deployed real-time system uses the Random Forest model with MediaPipe hand landmarks.
