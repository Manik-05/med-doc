import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Load the dataset (download from Kaggle if needed)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# Add a synthetic Year column assuming data spans from 2010-2024 for demonstration
df["Year"] = np.random.randint(2015, 2025, df.shape[0])

# Filter data from the last 5 years
current_year = datetime.now().year
df = df[df["Year"] >= (current_year - 5)]

# Feature Engineering: Creating BMI categories and age groups
df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese"])
df["Age_Group"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, np.inf], labels=["20-30", "30-40", "40-50", "50+"])

# Convert categorical features into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=["BMI_Category", "Age_Group"], drop_first=True)

# Drop the Year column before training
X = df.drop(["Outcome", "Year"], axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for user input and prediction
def predict_diabetes():
    print("Enter patient details:")
    user_data = {
        "Pregnancies": float(input("Pregnancies: ")), 
        "Glucose": float(input("Glucose Level: ")), 
        "BloodPressure": float(input("Blood Pressure: ")), 
        "SkinThickness": float(input("Skin Thickness: ")), 
        "Insulin": float(input("Insulin Level: ")), 
        "BMI": float(input("BMI: ")), 
        "DiabetesPedigreeFunction": float(input("Diabetes Pedigree Function: ")), 
        "Age": float(input("Age: "))
    }
    
    # Create DataFrame for user input
    user_df = pd.DataFrame([user_data])
    
    # Feature Engineering for user input
    user_df["BMI_Category"] = pd.cut(user_df["BMI"], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese"])
    user_df["Age_Group"] = pd.cut(user_df["Age"], bins=[20, 30, 40, 50, np.inf], labels=["20-30", "30-40", "40-50", "50+"])
    user_df = pd.get_dummies(user_df, columns=["BMI_Category", "Age_Group"], drop_first=True)
    
    # Align columns with training data
    for col in X.columns:
        if col not in user_df:
            user_df[col] = 0  # Add missing columns with default value
    user_df = user_df[X.columns]  # Reorder columns
    
    # Standardize input data
    user_input_scaled = scaler.transform(user_df)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)[0]
    risk = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"
    print(f"Prediction: {risk}")

# Run user input prediction
predict_diabetes()


# Enter patient details:
# Pregnancies: 2  
# Glucose Level: 145  
# Blood Pressure: 85  
# Skin Thickness: 20  
# Insulin Level: 90  
# BMI: 28.4  
# Diabetes Pedigree Function: 0.6  
# Age: 42  
