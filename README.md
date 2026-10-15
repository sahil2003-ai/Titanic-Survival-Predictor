# Titanic Survival Predictor

A supervised-learning portfolio project that explores passenger survival classification and exposes predictions through a Streamlit interface.

## Project files

| File | Purpose |
|---|---|
| `Titanic_train.csv` | Labeled training dataset |
| `Titanic_test.csv` | Test records without survival labels |
| `Titanic Survival Predictor.ipynb` | EDA, preprocessing, and model training |
| `app.py` | Streamlit application |
| `model.pkl` | Saved model artifact |
| `requirements.txt` | Python dependencies |
| `APP_LINK.md` | Deployment information, if current |

## Dataset metrics

Saved notebook output and the repository CSV files show:

| Metric | Training file | Test file |
|---|---:|---:|
| Rows | 891 | 418 |
| Columns | 12 | 11 |
| Target field | `Survived` | Not present |

In the training data, 342 passengers survived and 549 did not, giving a training-set survival proportion of **38.38%**. The training notebook reports missing values in Age (177), Cabin (687), and Embarked (2); the other listed fields have no missing values in the saved output.

## Modeling workflow

1. Inspect the training and test schemas.
2. Explore target balance and missing data.
3. Drop identifier/text fields selected by the notebook and impute missing Age values using the training median.
4. Encode categorical features and train the implemented classifier.
5. Save the model and expose inference through Streamlit.
6. Validate that the app uses the same feature order and preprocessing assumptions as training.

## Reported training-set performance

The saved notebook reports the following **in-sample training metrics**:

| Metric | Result |
|---|---:|
| Accuracy | 80.02% |
| Precision | 76.11% |
| Recall | 69.88% |
| F1-score | 72.87% |

**Important:** These are measurements on the same 891 rows used to fit the model, not a held-out validation or test score. They can overstate generalization and should not be presented as production performance. The 418-row test CSV has no target labels, so it cannot independently provide accuracy. A defensible next step is a stratified train/validation split or cross-validation on the labeled training data, followed by a single final refit if required.

The model workflow also calculates a ROC curve using training predictions; that AUC is not a held-out estimate and is not promoted here as such.

## Run locally

1. Create a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application from the repository root:

   ```bash
   streamlit run app.py
   ```

4. Check that the saved model loads and that input fields match the training feature schema.

Run the notebook separately in Jupyter if you want to reproduce preprocessing and training.

## Limitations

- Training metrics are in-sample only.
- Passenger survival is a historical classification exercise, not a causal analysis.
- Dataset source and license should be documented.
- The saved model and application should be tested together after any feature/preprocessing changes.

## Skills demonstrated

Python, pandas, exploratory data analysis, missing-value handling, categorical encoding, logistic/classification modeling as implemented, evaluation metrics, model serialization, and Streamlit.

## Author

Sahil Gaikwad · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
