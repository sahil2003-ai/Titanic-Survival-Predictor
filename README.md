# Titanic-Survival-Predictor

Titanic Survival Predictor is a machine learning web application that predicts the survival probability of a Titanic passenger using logistic regression based on real passenger data. The project includes all steps from data preparation and exploratory analysis to model building and a user-friendly Streamlit interface.

Features
Data cleaning and preprocessing of Titanic dataset

Exploratory Data Analysis with visualizations (pairplots, boxplots)

Feature encoding and handling missing values

Building and training a Logistic Regression model with scikit-learn

Model performance evaluation (accuracy, precision, recall, F1-score, ROC curve, feature coefficients)

Saving trained model for later inference

Interactive Streamlit app to input passenger details and get survival prediction with probability

Project Structure
Titanic-Survival-Predictor.ipynb — Jupyter notebook containing all EDA, feature engineering, model training, evaluation, and model serialization (model.pkl)

app.py — Streamlit app allowing users to predict survival by entering passenger details and viewing predicted outcome and probability

Installation
Requirements
Python 3.7+

Required Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

streamlit

joblib

Setup
Clone the repository

text
git clone https://github.com/yourusername/titanic-survival-predictor.git
cd titanic-survival-predictor
Install dependencies

text
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
Prepare Data

Place train.csv and test.csv datasets in the working directory if running the notebook.

Run all cells in Titanic-Survival-Predictor.ipynb to train, evaluate, and save the model as model.pkl.

Run the Web App

text
streamlit run app.py
Usage
Enter the passenger features in the Streamlit form:

Class (Pclass)

Sex

Age

Number of siblings/spouses (SibSp)

Number of parents/children (Parch)

Fare

Embarked location (S, C, Q)

The app instantly predicts:

Survival status ("Survived" or "Not Survived")

Survival probability

Model Details
Type: Logistic Regression (scikit-learn)

Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Encoding: LabelEncoder for categorical features, SimpleImputer for missing values

Performance:

Accuracy ~80%

Feature importance available in the notebook

License
This project is licensed under the MIT License.

Author
Sahil Gaikwad

