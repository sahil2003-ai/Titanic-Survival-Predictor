import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')  # Save your trained model as logistic_model.pkl

st.title('Titanic Survival Predictor')

pclass = st.number_input('Pclass', 1, 3, 1)
sex = st.selectbox('Sex', ['female', 'male'])
age = st.number_input('Age', 0, 100, 30)
sibsp = st.number_input('SibSp', 0, 10, 0)
parch = st.number_input('Parch', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 600.0, 30.0)
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

sex_map = {'female': 0, 'male': 1}
embarked_map = {'S': 2, 'C': 0, 'Q': 1}

input_data = np.array([[pclass, sex_map[sex], age, sibsp, parch, fare, embarked_map[embarked]]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0,1]

st.write('Survival Prediction:', 'Survived' if prediction == 1 else 'Not Survived')
st.write('Survival Probability:', round(probability,2))