# train_model.py
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv("sensor_with_broken.csv", header=None)
df = df[0].str.split(",", expand=True)  # split that one big string column
df.columns = df.iloc[0]                 # set column names
df = df[1:] 


df1 = df.drop(columns=['sensor_00', 'sensor_15', 'sensor_50', 'sensor_51'])
df1 = df1.fillna(df1.mean(numeric_only=True))

# Sensors
sensor_cols = [col for col in df1.columns if col.startswith("sensor_")]
df1[sensor_cols] = df1[sensor_cols].replace('', np.nan)  # Handle empty strings
df1[sensor_cols] = df1[sensor_cols].astype(float)
df1 = df1.fillna(df1.mean(numeric_only=True)) 

# FFT feature extraction
window_size = 128
step_size = 64
X_fft = []
y_fft = []

for start in range(0, len(df1) - window_size + 1, step_size):
    window = df1.iloc[start:start+window_size]
    features = {}
    for col in sensor_cols:
        signal = window[col].values
        signal -= np.mean(signal)
        fft_vals = np.fft.fft(signal)
        fft_mag = np.abs(fft_vals[:window_size // 2])
        features[f'{col}_fft_mean'] = np.mean(fft_mag)
        features[f'{col}_fft_std'] = np.std(fft_mag)
        features[f'{col}_fft_max'] = np.max(fft_mag)
        features[f'{col}_fft_freq_at_max'] = np.argmax(fft_mag)
    
    label_counts = Counter(window['machine_status'].str.strip().str.upper())
    total = sum(label_counts.values())

    broken_ratio = label_counts.get('BROKEN', 0) / total
    recovering_ratio = label_counts.get('RECOVERING', 0) / total

    # Label based on meaningful presence (threshold: 10%)
    # if broken_ratio >= 0.01:
    #     majority_label = 'BROKEN'
    # elif recovering_ratio >= 0.02:
    #     majority_label = 'RECOVERING'
    # else:
    #     majority_label = 'NORMAL'

    if recovering_ratio >= 0.01:
        majority_label = 'RECOVERING'
    elif broken_ratio >= 0.02:
        majority_label = 'BROKEN'
    else:
        majority_label = 'NORMAL'

    X_fft.append(features)
    y_fft.append(majority_label)




# Prepare final dataset
X_fft_df = pd.DataFrame(X_fft)
y_fft_series = pd.Series(y_fft, name="machine_status")
df_final = X_fft_df.copy()
df_final['machine_status'] = y_fft_series

# Encode labels
le = LabelEncoder()
df_final['label'] = le.fit_transform(df_final['machine_status'])
X = df_final.drop(columns=['machine_status', 'label'])
y = df_final['label']
print("Label classes:", le.classes_)
print(df_final['machine_status'].value_counts())



# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Metrics
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

joblib.dump(X.columns.tolist(), 'sensor_columns.pkl')

# Save model and label encoder
joblib.dump(clf, 'rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model and encoder saved.")
print(np.unique(y))  # y = your label array
print("Training class counts:")
print(y_train.value_counts())

