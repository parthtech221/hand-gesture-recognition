from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'dataset' / 'hand_gesture_master_dataset.csv'
OUT = ROOT / 'splits'
OUT.mkdir(exist_ok=True)

GESTURES = ['01_palm', '02_fist', '03_thumb', '04_index', '05_peace']
df = pd.read_csv(RAW)
# Remove OK gesture from the current project scope.
df = df[df['gesture'].isin(GESTURES)].copy()
df['label'] = df['gesture'].map(LABEL_MAP)

# One row group = one physical capture session (150 consecutive samples).
groups = df[['session_id', 'gesture']].drop_duplicates().sort_values('session_id')

train_groups, test_groups = train_test_split(
    groups,
    test_size=0.20,
    random_state=42,
    stratify=groups['gesture'],
)

train_ids = set(train_groups['session_id'])
test_ids = set(test_groups['session_id'])

train_df = df[df['session_id'].isin(train_ids)].copy()
test_df = df[df['session_id'].isin(test_ids)].copy()

train_df.to_csv(OUT / 'train_raw.csv', index=False)
test_df.to_csv(OUT / 'test_raw.csv', index=False)
train_groups.to_csv(OUT / 'train_sessions.csv', index=False)
test_groups.to_csv(OUT / 'test_sessions.csv', index=False)

print(f'Total samples : {len(df)}')
print(f'Train samples : {len(train_df)}')
print(f'Test samples  : {len(test_df)}')
print(f'Train sessions: {len(train_ids)}')
print(f'Test sessions : {len(test_ids)}')
print('\nTrain gesture counts:')
print(train_df['gesture'].value_counts().sort_index())
print('\nTest gesture counts:')
print(test_df['gesture'].value_counts().sort_index())
print('\nTrain/test session overlap:', bool(train_ids & test_ids))
