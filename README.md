# Titanic-Survival-Predictor

## Overview

Titanic Survival Predictor is a machine learning web application that predicts the survival probability of Titanic passengers using logistic regression. Built with Python, scikit-learn, and Streamlit, this project demonstrates a complete ML workflow from data exploration to interactive prediction interface.

## Features

✅ **Data Processing**
- Data cleaning and preprocessing of Titanic dataset
- Handling missing values with SimpleImputer
- Feature encoding using LabelEncoder for categorical variables

✅ **Exploratory Data Analysis**
- Comprehensive EDA with visualizations (pairplots, boxplots, heatmaps)
- Statistical analysis of passenger features
- Distribution analysis of target variable (survival)

✅ **Machine Learning Model**
- Logistic Regression classifier from scikit-learn
- Model training on Titanic training dataset
- Model performance evaluation (Accuracy, Precision, Recall, F1-Score, ROC Curve)
- Feature importance analysis

✅ **Web Application**
- Interactive Streamlit web app for real-time predictions
- User-friendly form to input passenger details
- Instant survival prediction with probability score
- Visual feedback with prediction status and probability bar chart

## Project Structure

```
Titanic-Survival-Predictor/
├── Titanic Survival Predictor.ipynb    # Complete ML pipeline notebook
├── app.py                              # Streamlit web application
├── model.pkl                           # Trained logistic regression model
├── Titanic_train.csv                   # Training dataset
├── Titanic_test.csv                    # Test dataset
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

### Key Files

- **Titanic Survival Predictor.ipynb**: Jupyter notebook containing:
  - Data loading and exploration
  - Feature engineering and preprocessing
  - Model training and evaluation
  - Model serialization (saved as model.pkl)

- **app.py**: Streamlit application featuring:
  - Input form for passenger details (Class, Sex, Age, SibSp, Parch, Fare, Embarked)
  - Real-time survival predictions
  - Probability visualization
  - Interactive user interface

## Installation

### Requirements

- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/sahil2003-ai/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor
```

2. **Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

3. **Prepare the data** (Optional - for retraining the model)

- Ensure `Titanic_train.csv` and `Titanic_test.csv` are in the working directory
- Run all cells in the Jupyter notebook to retrain and save the model

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `localhost:8501`

### Making Predictions

1. Fill in the passenger details in the Streamlit form:
   - **Class (Pclass)**: Passenger class (1, 2, or 3)
   - **Sex**: Gender (Male or Female)
   - **Age**: Passenger age (0-80)
   - **SibSp**: Number of siblings/spouses (0-10)
   - **Parch**: Number of parents/children (0-10)
   - **Fare**: Ticket fare (0.0-600.0)
   - **Embarked**: Port of embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)

2. Click the **"Predict Survival"** button

3. View the prediction results:
   - Survival status ("Survived" or "Not Survived")
   - Survival probability (0-100%)
   - Visual probability bar chart

## Model Details

### Algorithm
- **Type**: Logistic Regression (Binary Classification)
- **Library**: scikit-learn

### Features Used
- Pclass (Passenger Class)
- Sex (Categorical, encoded)
- Age
- SibSp (Siblings/Spouses)
- Parch (Parents/Children)
- Fare (Ticket Price)
- Embarked (Port, encoded)

### Preprocessing Pipeline
- **Categorical Encoding**: LabelEncoder transforms Sex and Embarked to numeric values
- **Missing Value Handling**: SimpleImputer fills missing Age and Embarked values

### Model Performance
- **Training Accuracy**: ~80%
- **Feature Importance**: Available in the Jupyter notebook
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Technologies Used

- **Python 3.7+**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **matplotlib & seaborn**: Data visualization
- **Streamlit**: Web application framework
- **joblib**: Model serialization

## How to Contribute

Contributions are welcome! Feel free to:
- Report bugs or suggest improvements
- Fork the repository and create pull requests
- Improve documentation or add new features

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Author

**Sahil Gaikwad** ([GitHub](https://github.com/sahil2003-ai))

## Acknowledgments

- Titanic dataset source: Kaggle
- Inspired by classic machine learning classification problems
- Built with modern Python ML tools and best practices

---

**Last Updated**: December 2025
