import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome')
X = diabetes_dataset.drop('Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit UI
st.title("Diabetes Prediction Web App")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=5)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=166)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=19)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=175)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.8)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.587)
age = st.number_input("Age", min_value=1, max_value=120, value=51)

if st.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1,-1)
    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)

    if prediction[0] == 1:
        st.success("The patient is **diabetic**")
    else:
        st.success("The patient is **not diabetic**")

