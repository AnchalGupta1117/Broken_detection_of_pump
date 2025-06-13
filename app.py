
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and encoder
sensor_cols_from_model = joblib.load("sensor_columns.pkl")  
model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")
drop_cols = ['sensor_00', 'sensor_15', 'sensor_50', 'sensor_51']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        df = pd.read_csv(file)
        if 'timestamp' not in df.columns:
            return jsonify({'error': 'Missing timestamp column'}), 400

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        original_df = df.copy()

        # Drop and clean
        if 'machine_status' in df.columns:
            df = df.drop(columns=['machine_status'])
        df = df.drop(columns=drop_cols, errors='ignore')

        raw_sensor_cols = sorted({name.split('_fft_')[0] for name in sensor_cols_from_model})

        df = df[['timestamp'] + raw_sensor_cols]

        df = df.fillna(df.mean(numeric_only=True))
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

        # FFT + Prediction
        window_size = 128
        step_size = 64
        timestamps = []
        X_fft = []

        for start in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[start:start + window_size]
            features = {}
            for col in sensor_cols:
                signal = window[col].values - np.mean(window[col].values)
                fft_vals = np.fft.fft(signal)
                fft_mag = np.abs(fft_vals[:window_size // 2])
                features[f'{col}_fft_mean'] = np.mean(fft_mag)
                features[f'{col}_fft_std'] = np.std(fft_mag)
                features[f'{col}_fft_max'] = np.max(fft_mag)
                features[f'{col}_fft_freq_at_max'] = np.argmax(fft_mag)
            X_fft.append(features)
            timestamps.append(window.iloc[0]['timestamp'])

        if not X_fft:
            return jsonify({'error': 'Not enough data for one FFT window'}), 400

        X_new = pd.DataFrame(X_fft)
        preds = model.predict(X_new)
        labels = le.inverse_transform(preds)

        print("Prediction label counts:")
        print(pd.Series(labels).value_counts())



        # Breakdown and Recovery detection
        breakdowns = []
        recoveries = []
        i = 0
        while i < len(labels):
            label = labels[i]
            if label != 'NORMAL':
                start_time = timestamps[i]
                duration = 1
                curr_label = label
                i += 1
                while i < len(labels) and labels[i] == curr_label:
                    duration += 1
                    i += 1
                end_index = i - 1 if i - 1 < len(timestamps) else len(timestamps) - 1
                end_time = timestamps[end_index]
                total_seconds = duration * window_size
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                record = {
                    'start': str(start_time),
                    'end': str(end_time),
                    'duration': f"{hours} hours {minutes} minutes"
                }

                if curr_label == 'BROKEN':
                    breakdowns.append(record)
                elif curr_label == 'RECOVERING':
                    recoveries.append(record)
            else:
                i += 1

        
        plot_urls = []
        fft_plot_urls = []

        for col in sensor_cols[:5]:  # Only first 5 sensors
            # Plot time-series
            plt.figure(figsize=(12, 4))
            plt.plot(original_df['timestamp'], original_df[col], label=col, color='blue')
            plt.xlabel('Time')
            plt.ylabel(col)
            plt.title(f'{col} Over Time')
            plt.tight_layout()
    
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
            plot_urls.append(f'data:image/png;base64,{plot_data}')
            
            # Plot FFT
            signal = original_df[col].values - np.mean(original_df[col].values)
            n = len(signal)
            yf = np.fft.fft(signal)
            xf = np.fft.fftfreq(n, 1.0)[:n // 2]  # Assuming 1 Hz sampling rate
            yf_abs = np.abs(yf[:n // 2])

            plt.figure(figsize=(12, 4))
            plt.plot(xf, yf_abs, color='red')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'{col} FFT Spectrum')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            fft_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
            fft_plot_urls.append(f'data:image/png;base64,{fft_plot_data}')

        print("Recoveries being sent to frontend:", recoveries)

        return jsonify({
            'breakdowns': breakdowns ,
            'recoveries': recoveries ,
            'message': 'No breakdown detected' if not breakdowns else None,
            'plot_urls': plot_urls,     
            'fft_urls': fft_plot_urls,
            'label_counts': pd.Series(labels).value_counts().to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
