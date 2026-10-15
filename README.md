# Titanic Survival Predictor

A machine-learning learning project that estimates Titanic passenger survival probability using a scikit-learn classification workflow and a Streamlit interface.

## Project Overview
The project covers data preparation, exploratory analysis, model training, evaluation, and interactive predictions from passenger attributes.

## Features Documented
- Missing-value handling with `SimpleImputer`
- Categorical feature encoding
- Exploratory plots and statistical summaries
- Logistic Regression classification
- Evaluation using accuracy, precision, recall, F1-score, and ROC analysis
- Streamlit interface for entering passenger attributes and viewing a prediction

## Technology
- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Streamlit

Confirm exact dependencies from the project files before installing.

## Run Locally
1. Clone this repository and enter its directory.
2. Create and activate a virtual environment.
3. Install the dependencies listed in the repository (add a `requirements.txt` if missing).
4. Check the notebook for the intended training and preprocessing workflow.
5. Run the app with `streamlit run app.py` if `app.py` is present and configured.

## Evaluation Checklist
- Split data before fitting preprocessing steps.
- Fit imputers and encoders on training data only.
- Compare with a simple baseline.
- Report metrics from a held-out test set.
- Check that the saved model and app use the same feature order and preprocessing.

## Skills Demonstrated
Classification, preprocessing, EDA, model evaluation, model serialization, and building a simple ML interface.

## Limitations
This is a historical educational dataset. Predictions are demonstrations of a model, not causal explanations of survival or a generalizable safety tool.

## Recommended Improvements
Add a reproducible requirements file, dataset source, model metrics, screenshots, and a clear explanation of the train/test split.