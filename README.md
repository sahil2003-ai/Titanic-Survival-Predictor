# Titanic Survival Predictor

A supervised-learning project exploring passenger survival classification, with preprocessing and model training in a notebook and prediction inference through a Streamlit interface.

![Python](https://img.shields.io/badge/Python-ML-blue) ![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)

## Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Workflow](#model-workflow)
- [Results and Metrics](#results-and-metrics)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Run the App](#run-the-app)
- [Limitations and Next Steps](#limitations-and-next-steps)

## Overview
The project explores which passenger attributes are associated with survival in the Titanic training dataset and packages a trained classifier for interactive predictions.

## Dataset
The repository includes `Titanic_train.csv` and `Titanic_test.csv`.

| Metric | Training file | Test file |
|---|---:|---:|
| Rows | 891 | 418 |
| Columns | 12 | 11 |
| Target field | `Survived` | Not present |

In the training data, **342 passengers survived** and **549 did not**, giving a training-set survival proportion of **38.38%**. Saved notebook output reports missing values in Age (177), Cabin (687), and Embarked (2); confirm these counts when rerunning the notebook.

## Model Workflow
1. Inspect train/test schemas and target balance.
2. Explore missing values and distributions.
3. Apply the notebook's preprocessing, including Age imputation and categorical encoding.
4. Train and evaluate the classifier.
5. Save the model artifact.
6. Use the Streamlit interface for inference, ensuring app inputs match the training feature schema.

## Results and Metrics
Verified data results: 891 labeled training rows, 418 unlabeled test rows, and a 38.38% survival proportion in the training set. A model accuracy or leaderboard score is not quoted here because a reproducible evaluation output was not confirmed in this review. Add the held-out metric with its split strategy after rerunning the notebook.

## Technology Stack
- Python and pandas
- Jupyter Notebook
- scikit-learn and model artifact tooling used in the notebook
- Streamlit

## Repository Structure
```text
Titanic-Survival-Predictor/
├── Titanic_train.csv
├── Titanic_test.csv
├── Titanic Survival Predictor.ipynb
├── app.py
├── model.pkl
├── requirements.txt
├── APP_LINK.md
└── README.md
```

## Run the App
```bash
git clone https://github.com/sahil2003-ai/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor
python -m venv .venv
```
Activate the environment, then install dependencies and launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Run the notebook separately to reproduce preprocessing and training. Confirm that the serialized model and app use identical feature names, order, and encoding.

## Limitations and Next Steps
- Document the source dataset and license.
- Report a reproducible holdout metric and baseline.
- Keep preprocessing identical between training and app inference.
- Avoid using the test file for model selection because its target is not provided.
- Add genuine app screenshots if stored in the repository.

## Skills Demonstrated
Classification workflow, data cleaning, missing-value handling, categorical encoding, model persistence, and Streamlit inference.

## Author
**Sahil Gaikwad** · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
