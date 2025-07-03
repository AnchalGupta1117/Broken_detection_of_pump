# 🔧 Pump Breakdown Predictor

This project predicts breakdowns and recovery phases of an industrial centrifugal pump using sensor data (like vibration, temperature, etc.) and a trained ML model.

## 🚀 Live App

👉 [Click here to try the app](https://web-production-bebef.up.railway.app)

---

## 📦 Overview

Given time-series sensor data, the app:
- Uses FFT to extract features
- Applies a trained RandomForest model
- Predicts machine status (`Healthy`, `Broken`, `Recovering`)
- Visualizes breakdown and recovery periods

---


## 🔧 Tech Stack

- **Frontend**: HTML/CSS/JS
- **Backend**: Flask (Python)
- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Deployment**: Railway (Docker-based)

---

## 📁 How to Use

1. Upload your `.csv` file containing sensor data
2. The backend extracts features & predicts status
3. The output includes breakdown/recovery segments & optional visualizations

---

## 📄 Example Input CSV Format

| timestamp        | vibration | flow_rate | temperature | ... |
|------------------|-----------|------------|-------------|-----|
| 2025-07-01 00:00 | 0.0035    | 1.26       | 35.2        | ... |
| 2025-07-01 00:01 | 0.0037    | 1.20       | 35.3        | ... |

---

## 📸 Screenshots

### 🏠 Home Page  
![Home Page](Screenshot1.png)

### 📂 File Upload  
![File Upload](Screenshot%202025-07-03%20220912.png)

### 📈 Prediction Output  
![Prediction Output](Screenshot%202025-07-03%20220937.png)

### ✅ Deployment Success  
![Deployment](Screenshot%202025-07-03%20220954.png)
![image](https://github.com/user-attachments/assets/f7f0fcd3-cb1f-4ba0-92c4-e57a1ebaf52c)
![image](https://github.com/user-attachments/assets/30e8ddf8-b29a-43ba-b031-a46ae2ff507f)
![image](https://github.com/user-attachments/assets/bfb89826-1e32-4293-984b-e6568b2bd7b4)
---

## 👩‍💻 Author

Made with ❤️ by [Anchal Gupta](https://github.com/AnchalGupta1117)


