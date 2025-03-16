import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the datasets
calories_data = pd.read_csv('data/calories.csv')
exercise_data = pd.read_csv('data/exercise.csv')

# Merge the datasets on the common key (assuming 'user_id')
merged_data = pd.merge(exercise_data, calories_data, on='user_id')

# Select relevant features
X = merged_data[['age', 'gender', 'bmi', 'duration', 'heart_rate', 'body_temp']]
y = merged_data['calories_burned']

# Encode gender (1 for Male, 0 for Female)
X.loc[:, 'gender'] = X['gender'].apply(lambda x: 1 if x == 'Male' else 0)


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and save SVM model
svm_model = SVR()
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'models/svm_model.pkl')

# Train and save Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
joblib.dump(logistic_regression_model, 'models/logistic_regression_model.pkl')

# Train and save Random Forest model
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)
joblib.dump(random_forest_model, 'models/random_forest_model.pkl')
