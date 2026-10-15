# Titanic Survival Predictor

A learning project that trains a classification model and provides an interactive Streamlit interface for exploring survival predictions on the historical Titanic dataset.

## Step-by-step workflow

1. Install dependencies from `requirements.txt` in a virtual environment.
2. Inspect `Titanic_train.csv` and `Titanic_test.csv`; identify the target, input features, and missing values.
3. Open `Titanic Survival Predictor.ipynb` and run the data exploration and cleaning cells.
4. Fit imputers and encoders on training data only to avoid leakage.
5. Train the Logistic Regression model and compare it with a simple baseline.
6. Evaluate with a held-out set using accuracy, precision, recall, F1-score, and ROC-AUC where appropriate.
7. Confirm the saved `model.pkl` and `app.py` use matching preprocessing and feature order.
8. Run `streamlit run app.py` and test a few inputs.
9. Document the actual metrics, split strategy, and limitations.

## Tools
Python, pandas, scikit-learn, Streamlit, Jupyter Notebook.

## Screenshots
No image files were found in the repository root. Add an app screenshot in `images/` and embed it here.

## Limitations
This is a historical educational dataset; model predictions are not causal explanations and may not generalize beyond the dataset.

## Next improvements
Add a screenshot, model metrics from a reproducible test set, and exact setup instructions.