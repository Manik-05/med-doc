import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


Input_data = pd.read_csv("iot_dataset.csv")
Input_data

Input_data.head()

Input_data.tail()

Input_data.tail()

Input_data.isnull().sum()

Input_data['Target'].value_counts()

Input_data.shape

plt.bar(Input_data['Patient ID'],Input_data['Temperature Data']) 
plt.title("Bar Chart of Temperature Data ") 
plt.xlabel('Patient ID')
plt.ylabel('Temperature Data')
plt.show()

plt.bar(Input_data['Patient ID'],Input_data['ECG Data']) 
plt.title("Bar Chart of ECG Data") 
plt.xlabel('Patient ID')
plt.ylabel('ECG Data')
plt.show()

plt.bar(Input_data['Patient ID'],Input_data['Pressure Data']) 
plt.title("Bar Chart of Pressure Data") 
plt.xlabel('Patient ID')
plt.ylabel('Pressure Data')
plt.show()

plt.bar(Input_data['Patient ID'],Input_data['Target']) 
plt.title("Bar Chart of Targeted Data") 
plt.xlabel('Patient ID')
plt.ylabel('Targeted Data')
plt.show()

plt.hist(Input_data['Patient ID'])
plt.title("Histogram of Patient ID")
plt.show()

plt.hist(Input_data['Temperature Data'])
plt.title("Histogram of Temperature Data")
plt.show()

plt.hist(Input_data['ECG Data'])
plt.title("Histogram of ECG Data")
plt.show()

plt.hist(Input_data['Pressure Data'])
plt.title("Histogram of Pressure Data")
plt.show()

# count plot on single categorical variable
sns.countplot(x ='Pressure Data', data = Input_data)
 
# Show the plot
plt.show()

# count plot on single categorical variable
sns.countplot(x ='ECG Data', data = Input_data)
 
# Show the plot
plt.show()

# count plot on single categorical variable
sns.countplot(x ='Temperature Data', data = Input_data)
 
# Show the plot
plt.show()

# count plot on single categorical variable
sns.countplot(x ='Patient ID', data = Input_data)
 
# Show the plot
plt.show()

sns.kdeplot(Input_data['Patient ID'])

sns.kdeplot(Input_data['Temperature Data'])

sns.kdeplot(Input_data['ECG Data'])

sns.kdeplot(Input_data['Pressure Data'])

sns.kdeplot(Input_data['Target'])

import seaborn as sns
corr = Input_data.corr()
plt.subplots(figsize=(5,5))
sns.heatmap(corr, annot = True)

X = Input_data.drop('Target',axis=1)
X

Y = Input_data['Target']
Y 

from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1 =  train_test_split(X,Y,random_state=42,test_size=0.2,shuffle=True)

from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics
NB_Algorithm = MultinomialNB()
NB_Algorithm.fit(x_train1, y_train1)
NB_Algorithm_Prediction = NB_Algorithm.predict(x_test1)
Accuracy_NB = metrics.accuracy_score(y_test1, NB_Algorithm_Prediction)
print('Accuracy of Naive Bayes Algorithm', Accuracy_NB)

NB_Algorithm_Prediction

from sklearn.metrics import classification_report, confusion_matrix
CM_NB=confusion_matrix(y_test1, NB_Algorithm_Prediction)
sns.heatmap(CM_NB, annot=True, fmt='d', cmap='YlGnBu')
print(classification_report(y_test1, NB_Algorithm_Prediction))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
DT_Algorithm = DecisionTreeClassifier()
DT_Algorithm.fit(x_train1, y_train1)
DT_Algorithm_Prediction = DT_Algorithm.predict(x_test1)
Accuracy_DT = accuracy_score(y_test1, DT_Algorithm_Prediction)
print('Accuracy of Decision Tree Algorithm', Accuracy_DT)

from sklearn.metrics import classification_report, confusion_matrix
CM_DT=confusion_matrix(y_test1, DT_Algorithm_Prediction)
sns.heatmap(CM_DT, annot=True, fmt='d', cmap='YlGnBu')
print(classification_report(y_test1, DT_Algorithm_Prediction))

# LOgistic Regression algo

from sklearn.linear_model import LogisticRegression
LR_Algorithm = LogisticRegression()
LR_Algorithm.fit(x_train1, y_train1)
LR_Algorithm_Prediction = LR_Algorithm.predict(x_test1)
Accuracy_LR = accuracy_score(y_test1, LR_Algorithm_Prediction)
print('Accuracy of Logistic Regression Algorithm', Accuracy_LR)

LR_Algorithm_Prediction

from sklearn.metrics import classification_report, confusion_matrix
CM_LR=confusion_matrix(y_test1, LR_Algorithm_Prediction)
sns.heatmap(CM_LR, annot=True, fmt='d', cmap='YlGnBu')
print(classification_report(y_test1, LR_Algorithm_Prediction))

from sklearn.svm import SVC
SVM_Algorithm = SVC()
SVM_Algorithm.fit(x_train1, y_train1)
SVM_Algorithm_Prediction = SVM_Algorithm.predict(x_test1)
Accuracy_SVM = accuracy_score(y_test1, LR_Algorithm_Prediction)
print('Accuracy of Support Vector Machine Algorithm', Accuracy_SVM)

SVM_Algorithm_Prediction

from sklearn.metrics import classification_report, confusion_matrix
CM_SVM=confusion_matrix(y_test1, SVM_Algorithm_Prediction)
sns.heatmap(CM_SVM, annot=True, fmt='d', cmap='YlGnBu')
print(classification_report(y_test1, SVM_Algorithm_Prediction))


model_accuracy = pd.Series(data=[Accuracy_NB,Accuracy_DT,Accuracy_LR,Accuracy_SVM], 
                index=['Naive Bayes','Decision Tree','Logistic Regression','Support Vector Machine'])
fig= plt.figure(figsize=(5,5))
model_accuracy.sort_values().plot.barh()
plt.title('Comparison Graph of all the Algorithm')


# Final_Prediction_data = (71,9,32,0,77)
# Final_Prediction_data = np.array(Final_Prediction_data)
# Final_Prediction_data = Final_Prediction_data.reshape(1,-1)
# Final_prediction = DT_Algorithm.predict(Final_Prediction_data)

# if Final_prediction == 0:
#     print("The Patient Condition is Low")
# elif Final_prediction == 1:
#     print("The Patient Condition is Medium")
# else:
#     print("The Patient Condition is High")


# Final_Prediction_data = (10,2,32,0,77)
# Final_Prediction_data = np.array(Final_Prediction_data)
# Final_Prediction_data = Final_Prediction_data.reshape(1,-1)
# Final_prediction = DT_Algorithm.predict(Final_Prediction_data)

# if Final_prediction == 0:
#     print("The Patient Condition is Low")
# elif Final_prediction == 1:
#     print("The Patient Condition is Medium")
# else:
#     print("The Patient Condition is High")


# Final_Prediction_data = (43,1,32,0,0)
# Final_Prediction_data = np.array(Final_Prediction_data)
# Final_Prediction_data = Final_Prediction_data.reshape(1,-1)
# Final_prediction = DT_Algorithm.predict(Final_Prediction_data)

# if Final_prediction == 0:
#     print("The Patient Condition is Low")
# elif Final_prediction == 1:
#     print("The Patient Condition is Medium")
# else:
#     print("The Patient Condition is High")

import numpy as np

# Assuming DT_Algorithm is already trained
# Function to get user input and make a prediction
def predict_patient_condition():
    try:
        # Take input from the user
        input_data = input("Enter five values for  Patient ID,Temperature Data,ECG Data,Pressure Data,Target separated by commas (e.g. 71,9,32,0,77):- ")
        
        # Convert input string to a tuple of integers
        Final_Prediction_data = tuple(map(int, input_data.split(',')))

        # Convert to NumPy array and reshape
        Final_Prediction_data = np.array(Final_Prediction_data).reshape(1, -1)

        # Make a prediction
        Final_prediction = DT_Algorithm.predict(Final_Prediction_data)

        # Interpret the result
        if Final_prediction == 0:
            print("The Patient Condition is Low")
        elif Final_prediction == 1:
            print("The Patient Condition is Medium")
        else:
            print("The Patient Condition is High")
    
    except ValueError:
        print("Invalid input! Please enter five numbers separated by commas.")

# Call the function multiple times or use a loop if needed
predict_patient_condition()

# 'Patient ID'
# 'Temperature Data
# 'ECG Data'
# 'Pressure Data'
# 'Target'
