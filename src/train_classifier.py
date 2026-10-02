from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent

SPLITS = ROOT / 'splits'
RESULTS = ROOT / 'results'
MODELS = ROOT / 'models'
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

GESTURES = ['01_palm', '02_fist', '03_thumb', '04_index', '05_peace']
LABEL_MAP = {g: i for i, g in enumerate(GESTURES)}

# Use the same 63 landmark features produced by preprocessing.
LANDMARK_COLS = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]


def preprocess(df):
    xyz = df[LANDMARK_COLS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)

    # Wrist-centered coordinates.
    xyz = xyz - xyz[:, 0:1, :]

    # Normalize hand size using wrist -> middle MCP (landmark 9).
    scale = np.linalg.norm(xyz[:, 9, :], axis=1)
    scale[scale < 1e-8] = 1.0
    xyz = xyz / scale[:, None, None]

    # Canonicalize handedness: mirror X for left hands.
    left = df['handedness'].astype(str).str.lower().eq('left').to_numpy()
    xyz[left, :, 0] *= -1.0

    return xyz.reshape(-1, 63)

train = pd.read_csv(SPLITS / 'train_raw.csv')
test = pd.read_csv(SPLITS / 'test_raw.csv')

X_train = preprocess(train)
X_test = preprocess(test)
y_train = train['gesture'].map(LABEL_MAP).to_numpy()
y_test = test['gesture'].map(LABEL_MAP).to_numpy()

# Primary baseline: Random Forest.
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

# Secondary baseline: RBF SVM. Useful for compact landmark datasets.
svm = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm)

# Save the Random Forest as the first deployed model.
joblib.dump(rf, MODELS / 'gesture_random_forest.joblib')
joblib.dump(svm, MODELS / 'gesture_svm.joblib')

# Save test predictions and metrics.
predictions = test[['sample_id', 'session_id', 'person', 'lighting', 'gesture']].copy()
predictions['true_label'] = y_test
predictions['rf_prediction'] = pred_rf
predictions['rf_confidence'] = rf.predict_proba(X_test).max(axis=1)
predictions['svm_prediction'] = pred_svm
predictions['svm_confidence'] = svm.predict_proba(X_test).max(axis=1)
predictions.to_csv(RESULTS / 'test_predictions.csv', index=False)

rf_cm = confusion_matrix(y_test, pred_rf, labels=range(len(GESTURES)))
svm_cm = confusion_matrix(y_test, pred_svm, labels=range(len(GESTURES)))

metrics = {
    'dataset': {
        'total_samples': int(len(train) + len(test)),
        'train_samples': int(len(train)),
        'test_samples': int(len(test)),
        'train_sessions': int(train['session_id'].nunique()),
        'test_sessions': int(test['session_id'].nunique()),
    },
    'random_forest': {
        'accuracy': float(acc_rf),
        'classification_report': classification_report(
            y_test, pred_rf, target_names=GESTURES, output_dict=True
        ),
        'confusion_matrix': rf_cm.tolist(),
    },
    'svm_rbf': {
        'accuracy': float(acc_svm),
        'classification_report': classification_report(
            y_test, pred_svm, target_names=GESTURES, output_dict=True
        ),
        'confusion_matrix': svm_cm.tolist(),
    },
}

with open(RESULTS / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'Random Forest accuracy: {acc_rf:.4f}')
print(f'RBF SVM accuracy      : {acc_svm:.4f}')
print('\nRandom Forest classification report:')
print(classification_report(y_test, pred_rf, target_names=GESTURES, digits=4))
print('Random Forest confusion matrix:')
print(rf_cm)
print('\nRBF SVM classification report:')
print(classification_report(y_test, pred_svm, target_names=GESTURES, digits=4))
print('RBF SVM confusion matrix:')
print(svm_cm)
