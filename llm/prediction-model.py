import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import joblib # Import joblib for saving models

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# Add a synthetic Year column and filter data (as in your original script)
df["Year"] = np.random.randint(2015, 2025, df.shape[0])
current_year = datetime.now().year
df = df[df["Year"] >= (current_year - 5)]

# Feature Engineering
df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese"])
df["Age_Group"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, np.inf], labels=["20-30", "30-40", "40-50", "50+"])
df = pd.get_dummies(df, columns=["BMI_Category", "Age_Group"], drop_first=True)

# Prepare data for training
X = df.drop(["Outcome", "Year"], axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'diabetes_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and Scaler saved!")