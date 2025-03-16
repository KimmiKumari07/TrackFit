# 🔥 TrackFit(Personal Fitness Tracker)

Welcome to the **Personal Fitness Tracker**! This Streamlit-based web application allows users to predict the calories burned during exercise based on their personal details. The app leverages machine learning models to provide accurate predictions and helps users track their fitness journey over time.

## 🖥️ Screenshots

### Home Page
![Home Page](https://github.com/KimmiKumari07/TrackFit/blob/main/asserts/snap%201.png?raw=true)


### Prediction Result
![Prediction](https://github.com/KimmiKumari07/TrackFit/blob/main/asserts/snap%202.png?raw=true)

---

## 🚀 Features
- Input BMI directly or calculate it from height and weight.
- Predict calories burned using **SVM, Logistic Regression, or Random Forest** models.
- Automatically selects the best model based on user parameters.
- Generates a unique **User ID** for tracking predictions.
- Stores and displays **past predictions**.
- Allows downloading **past predictions as CSV**.
- Provides **historical data visualization**.
- Displays **feature importance** for Random Forest models.

---

## 📦 Installation Guide

### 1️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Ensure Model Files Are in Place
Make sure you have the pre-trained model files in the `models/` directory:
```
models/
├── svm_model.pkl
├── logistic_regression_model.pkl
└── random_forest_model.pkl
```

If these files are missing, retrain and save models using `joblib` before running the app.

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🛠 Technologies Used
- **Python**
- **Streamlit** (for UI and interactivity)
- **Pandas** (for data processing)
- **Scikit-Learn** (for machine learning models)
- **Joblib** (for model storage)
- **Matplotlib & Seaborn** (for visualizations)

---

## 📊 Data & Model Details
- The app supports three machine learning models: **Support Vector Machine (SVM)**, **Logistic Regression**, and **Random Forest**.
- Based on **BMI, age, and workout duration**, the app selects the most appropriate model.
- Predictions are stored in `data/history.csv` for tracking and analysis.

---

## 🏋️‍♂️ How It Works
1. **Enter personal details** such as BMI, age, gender, workout duration, heart rate, and body temperature.
2. **App selects the best model** automatically.
3. **Click "Start Prediction"** to get the estimated calories burned.
4. **Predictions are saved**, allowing users to track progress.
5. **Download historical predictions as CSV** for further analysis.

---

## 📄 License
This project is licensed under the **MIT License**.

