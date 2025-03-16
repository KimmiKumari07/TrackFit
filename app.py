import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import random
import string
import os

# Load the models
svm_model = joblib.load('models/svm_model.pkl')
logistic_regression_model = joblib.load('models/logistic_regression_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')

# Streamlit app
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🔥", layout="wide")

# Header and Description
st.title('🔥 Personal Fitness Tracker 🔥')
st.markdown("""Welcome to the Personal Fitness Tracker! This web app allows you to predict the calories burned during exercise based on your personal parameters. Input your details in the sidebar to get started. Track your fitness journey with historical data and insights from machine learning models.""")

# Sidebar for input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    input_option = st.sidebar.radio('Choose Input Option', ['Input BMI', 'Input Height and Weight'])
    
    if input_option == 'Input BMI':
        bmi = st.sidebar.slider('BMI', 0.0, 50.0, 22.0)
    else:
        height = st.sidebar.number_input('Height (cm)', min_value=50.0, max_value=250.0, value=170.0)
        weight = st.sidebar.number_input('Weight (kg)', min_value=5.0, max_value=200.0, value=70.0)
        bmi = weight / ((height / 100) ** 2)
        bmi = round(bmi, 3)
        st.sidebar.write(f'Calculated BMI: {bmi:.3f}')
    
    age = st.sidebar.slider('Age', 10, 100, 25)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    duration = st.sidebar.slider('Duration (mins)', 0, 300, 30)
    heart_rate = st.sidebar.slider('Heart Rate (bpm)', 30, 200, 70)
    body_temp = st.sidebar.slider('Body Temperature (°C)', 35.0, 50.0, 37.0)
    
    # Gender encoding
    gender = 1 if gender == 'Male' else 0

    data = {'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Duration': duration,
            'Heart Rate': heart_rate,
            'Body Temperature': body_temp}
    features = pd.DataFrame(data, index=[0])
    return features

def generate_user_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

input_df = user_input_features()

# Display input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
st.subheader('Model Prediction')

# Define thresholds for model selection
def select_model(bmi, duration, age):
    if bmi < 18.5:
        return 'Logistic Regression'
    elif bmi < 24.9:
        if duration < 60:
            return 'Logistic Regression'
        elif age < 30:
            return 'SVM'
        else:
            return 'Random Forest'
    else:
        if duration < 60:
            return 'Logistic Regression'
        elif age < 50:
            return 'SVM'
        else:
            return 'Random Forest'

# Automatically choose the model
model_choice = select_model(input_df['BMI'][0], input_df['Duration'][0], input_df['Age'][0])
st.write(f"Automatically selected model: {model_choice}")

# Generate unique user ID
user_id = generate_user_id()
st.write(f"User ID: {user_id}")

# Buttons to start and stop predictions
col1, col2 = st.columns([1, 1])
with col1:
    start_button = st.button('Start Prediction')
with col2:
    stop_button = st.button('Stop Prediction')

if start_button:
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_df)

    if model_choice == 'SVM':
        prediction = svm_model.predict(input_data)[0]
    elif model_choice == 'Logistic Regression':
        prediction = logistic_regression_model.predict(input_data)[0]
    elif model_choice == 'Random Forest':
        prediction = random_forest_model.predict(input_data)[0]

    st.metric(label="Predicted Calories Burned", value=f"{prediction:.2f} cal")

    # Store past predictions in session state
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []

    st.session_state['predictions'].append({
        'User ID': user_id,
        'Age': input_df['Age'][0],
        'Gender': 'Male' if input_df['Gender'][0] == 1 else 'Female',
        'BMI': input_df['BMI'][0],
        'Duration': input_df['Duration'][0],
        'Heart Rate': input_df['Heart Rate'][0],
        'Body Temperature': input_df['Body Temperature'][0],
        'Model': model_choice,
        'Prediction': prediction
    })

    # Define the history file path
    history_file_path = 'data/history.csv'

    # Ensure the folder exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the prediction data to the history CSV file
    history_data = {
        'User ID': user_id,
        'Age': input_df['Age'][0],
        'Gender': 'Male' if input_df['Gender'][0] == 1 else 'Female',
        'BMI': input_df['BMI'][0],
        'Duration': input_df['Duration'][0],
        'Heart Rate': input_df['Heart Rate'][0],
        'Body Temperature': input_df['Body Temperature'][0],
        'Model': model_choice,
        'Prediction': prediction
    }
    history_df = pd.DataFrame([history_data])

    # Append to history CSV if it exists, or create it if it doesn't
    history_df.to_csv(history_file_path, mode='a', header=not os.path.exists(history_file_path), index=False)

    st.success("Prediction completed successfully!")
elif stop_button:
    st.warning("Prediction stopped.")

# Display past predictions from session state
st.subheader('Past Predictions')
past_predictions = pd.DataFrame(st.session_state.get('predictions', []))
st.write(past_predictions)

# Data Export
if not past_predictions.empty:
    csv = past_predictions.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Past Predictions as CSV", data=csv, file_name='past_predictions.csv', mime='text/csv')

# Historical Data Visualization
if not past_predictions.empty:
    st.subheader('Historical Data Visualization')
    st.line_chart(past_predictions[['Age', 'Prediction']])



# Additional features
if st.checkbox('Show Feature Importance (Random Forest)'):
    if model_choice == 'Random Forest':
        importance = random_forest_model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': input_df.columns, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))
    else:
        st.write('Feature importance is only available for the Random Forest model.')
