from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')
X_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category_Normal', 'BMI_Category_Obese', 'BMI_Category_Overweight', 'BMI_Category_Underweight', 'Age_Group_30-40', 'Age_Group_40-50', 'Age_Group_50+', 'Age_Group_20-30'] # Ensure column order is correct

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_data = {
            "Pregnancies": float(data['Pregnancies']),
            "Glucose": float(data['Glucose']),
            "BloodPressure": float(data['BloodPressure']),
            "SkinThickness": float(data['SkinThickness']),
            "Insulin": float(data['Insulin']),
            "BMI": float(data['BMI']),
            "DiabetesPedigreeFunction": float(data['DiabetesPedigreeFunction']),
            "Age": float(data['Age'])
        }

        user_df = pd.DataFrame([user_data])

        # Feature Engineering for user input (same as in training)
        user_df["BMI_Category"] = pd.cut(user_df["BMI"], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=["Underweight", "Normal", "Overweight", "Obese"])
        user_df["Age_Group"] = pd.cut(user_df["Age"], bins=[20, 30, 40, 50, np.inf], labels=["20-30", "30-40", "40-50", "50+"])
        user_df = pd.get_dummies(user_df, columns=["BMI_Category", "Age_Group"], drop_first=True)

        # Align columns with training data
        for col in X_columns: # Using predefined column list
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[X_columns] # Reorder columns


        # Standardize input data
        user_input_scaled = scaler.transform(user_df)

        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        risk = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"

        return jsonify({'prediction': risk})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for development