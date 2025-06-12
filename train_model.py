import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df_raw = pd.read_csv("sensor_with_broken.csv", header=None)
df_raw = df_raw[0].str.split(",", expand=True)
df_raw.columns = df_raw.iloc[0]
df = df_raw[1:]

drop_cols = ['sensor_00', 'sensor_15', 'sensor_50', 'sensor_51']
df = df.drop(columns=drop_cols, errors='ignore')

sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
df[sensor_cols] = df[sensor_cols].replace('', np.nan).astype(float)
df = df.fillna(df.mean(numeric_only=True))

window_size = 128
step_size = 64
X_fft = []
y_fft = []

for start in range(0, len(df) - window_size + 1, step_size):
    window = df.iloc[start:start+window_size]
    features = {}

    for col in sensor_cols:
        signal = window[col].values - np.mean(window[col].values)
        fft_vals = np.fft.fft(signal)
        fft_mag = np.abs(fft_vals[:window_size // 2])

        features[f'{col}_fft_mean'] = np.mean(fft_mag)
        features[f'{col}_fft_std'] = np.std(fft_mag)
        features[f'{col}_fft_max'] = np.max(fft_mag)
        features[f'{col}_fft_freq_at_max'] = np.argmax(fft_mag)

    label_counts = Counter(window['machine_status'].str.strip().str.upper())
    total = sum(label_counts.values())

    if label_counts.get('BROKEN', 0) / total >= 0.03:
        label = 'BROKEN'
    elif label_counts.get('RECOVERING', 0) / total >= 0.02:
        label = 'RECOVERING'
    else:
        label = 'NORMAL'

    X_fft.append(features)
    y_fft.append(label)

X_df = pd.DataFrame(X_fft)
y_series = pd.Series(y_fft, name='machine_status')
le = LabelEncoder()
y_encoded = le.fit_transform(y_series)

print("Label classes:", le.classes_)
print("Class distribution:\n", y_series.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


joblib.dump(model, 'rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_df.columns.tolist(), 'sensor_columns.pkl')

print("✅ Model and encoder saved successfully.")
print("Training label counts:\n", pd.Series(y_train).value_counts())
