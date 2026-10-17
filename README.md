# Titanic Survival Predictor

A supervised machine-learning project that explores passenger survival patterns from the Titanic dataset, trains a **Logistic Regression** classifier, and exposes predictions through an interactive **Streamlit** application.

<p align="left">
  <img src="https://img.shields.io/badge/Python-Machine%20Learning-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-6A5ACD" alt="Logistic Regression" />
  <img src="https://img.shields.io/badge/Library-scikit--learn-F7931E" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

## Live application

**[Open the Titanic Survival Predictor](https://titanic-survival-predictor-zvf5nhdj2vpmdcuaqbtwaz.streamlit.app/)**

Enter passenger characteristics and get a predicted survival class and the model's estimated survival probability. This is an educational demonstration based on historical records, not a guarantee of any outcome.

## Project overview

This project uses passenger information to explore classification and the relationship between survival labels and features such as passenger class, sex, age, fare, family aboard and embarkation port. It combines exploratory data analysis (EDA), missing-value handling, categorical encoding, model training, evaluation and a lightweight prediction interface.

## Dataset

The repository includes the labeled training file and the unlabeled competition-style test file.

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| [`Titanic_train.csv`](Titanic_train.csv) | 891 | 12 | Labeled data containing `Survived` |
| [`Titanic_test.csv`](Titanic_test.csv) | 418 | 11 | Unlabeled data for prediction |

### Training set summary

| Metric | Result |
|---|---:|
| Survivors (`Survived = 1`) | 342 |
| Non-survivors (`Survived = 0`) | 549 |
| Training survival proportion | **38.38%** |
| Missing Age values | 177 |
| Missing Cabin values | 687 |
| Missing Embarked values | 2 |
| Average age among non-missing rows | 29.70 years |
| Median age among non-missing rows | 28 years |
| Average fare | 32.20 (currency not specified in repository) |

The counts above come from the notebook's saved outputs. The unlabeled test file has no `Survived` target, so it cannot be used to calculate test accuracy without ground-truth labels.

### Feature guide

| Feature | Description |
|---|---|
| `Survived` | Target: 0 = did not survive, 1 = survived |
| `Pclass` | Ticket/passenger class |
| `Sex` | Passenger sex as recorded in the dataset |
| `Age` | Passenger age |
| `SibSp` | Siblings/spouses aboard |
| `Parch` | Parents/children aboard |
| `Fare` | Recorded ticket fare |
| `Embarked` | Embarkation port code |

The notebook drops `PassengerId`, `Name`, `Ticket` and `Cabin` from the feature set, then imputes missing ages and embarkation values.

## Data preparation

The notebook performs the following steps:

1. Load `Titanic_train.csv` and `Titanic_test.csv`.
2. Inspect data types, summary statistics, class balance and missing values.
3. Remove `PassengerId`, `Name`, `Ticket` and `Cabin`.
4. Impute missing `Age` values using the training median.
5. Impute missing `Embarked` values using the most frequent training category.
6. Fill missing test-set `Fare` values with the test-set median when needed.
7. Encode `Sex` and `Embarked`.
8. Fit a Logistic Regression classifier and save the model to `model.pkl`.

## Model and evaluation

The notebook trains **Logistic Regression** with `max_iter=200`. It reports training-set metrics for the 891 labeled training records.

### Saved training-set results

| Metric | Result |
|---|---:|
| Training accuracy | **80.02%** |
| Training precision (positive class) | 76.11% |
| Training recall (positive class) | 69.88% |
| Training F1-score (positive class) | 72.87% |
| Training ROC-AUC | Calculated in notebook; numeric output not saved in the reviewed cells |

These are **in-sample training metrics**, not independent test-set or cross-validation performance. They can be optimistic because the model is evaluated on the same rows used for fitting. The repository's `Titanic_test.csv` has no labels, so a true test accuracy cannot be calculated from that file alone.

### Recommended next evaluation

For a stronger, portfolio-ready evaluation, split the labeled training data into stratified training and validation subsets, fit all preprocessing only on the training subset, and report:

- Accuracy and majority-class baseline
- Precision, recall and F1-score
- ROC-AUC
- Confusion matrix
- Cross-validation mean and variability

Use a held-out validation split for model evaluation; reserve the unlabeled test file for inference or an external leaderboard that provides ground-truth feedback.

## Exploratory data analysis

The notebook generates visualizations including:

- Age distribution histogram
- Fare distribution by survival label
- Pairwise feature plots for survival, passenger class, age, fare and family-size fields
- ROC curve based on the model's predictions for training data
- Logistic Regression coefficients for the selected features

### Coefficient interpretation

The saved coefficient output contains these values:

| Feature | Coefficient |
|---|---:|
| Sex (encoded) | -2.6311 |
| Pclass | -1.0507 |
| SibSp | -0.3162 |
| Embarked (encoded) | -0.2139 |
| Parch | -0.0848 |
| Age | -0.0380 |
| Fare | 0.0022 |

Because `Sex` and `Embarked` are label-encoded, the coefficient signs depend on their numeric encoding. Treat these as model coefficients, not causal effects. For categorical variables, one-hot encoding would generally yield more interpretable category-specific coefficients.

## Streamlit application

The app in [`app.py`](app.py) provides input controls for:

- Passenger class
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Fare
- Embarkation port

On submission, it displays the predicted class and the model's predicted survival probability, plus a progress bar and visual probability bar.

### Important consistency note

The notebook fits `LabelEncoder` for `Sex` and `Embarked`, but the app currently uses manually specified numeric mappings. These mappings need to match the exact fitted encoder classes and feature order used during training. In addition, the app loads a pickled Logistic Regression model and passes manually assembled feature values. Verify this mapping against the notebook before relying on predictions.

## Technology stack

| Area | Tools |
|---|---|
| Language and data | Python, pandas, NumPy |
| EDA and visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn Logistic Regression |
| Preprocessing | SimpleImputer, LabelEncoder |
| Model serialization | pickle / joblib-compatible model artifact |
| Interactive app | Streamlit |

Dependencies are listed in [`requirements.txt`](requirements.txt).

## Repository structure

```text
Titanic-Survival-Predictor/
├── Titanic_train.csv
├── Titanic_test.csv
├── Titanic Survival Predictor.ipynb
├── app.py
├── model.pkl
├── requirements.txt
├── APP_LINK.md
├── LICENSE
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/sahil2003-ai/Titanic-Survival-Predictor.git
cd Titanic-Survival-Predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Run it from the repository root so `model.pkl` can be found. To reproduce the analysis, open [`Titanic Survival Predictor.ipynb`](Titanic%20Survival%20Predictor.ipynb) and run the cells in order.

## Images and screenshots

The repository currently contains the notebook, CSV files, app code and model artifact, but no separate `images/` directory or exported chart screenshots. The notebook generates charts as cell outputs, so this README does not embed invented images. To show genuine visuals on the GitHub landing page, export the age distribution, fare-by-survival plot, ROC curve and an actual app screenshot into an `images/` folder, then reference those files here.

## Limitations and next steps

- **Independent evaluation:** add a stratified validation split or cross-validation; do not present training accuracy as test performance.
- **Preprocessing consistency:** store imputers and encoders with the model in a single scikit-learn Pipeline.
- **Categorical encoding:** use one-hot encoding for nominal fields such as `Embarked` to avoid implying an artificial order.
- **App validation:** verify the app's manual mappings and input ordering against the notebook's fitted encoders and training features.
- **Test-set target:** the supplied test data does not include `Survived`; do not claim test accuracy using it.
- **Dataset provenance:** document the source and license if redistributing the data.
- **Visual documentation:** export actual notebook charts and app screenshots into `images/`.
- **Responsible interpretation:** historical classification patterns are not causal explanations of survival.

## Skills demonstrated

Data cleaning · exploratory data analysis · missing-value imputation · categorical encoding · Logistic Regression · classification metrics · ROC analysis · coefficient interpretation · model serialization · Streamlit application development

## Author

**Sahil Gaikwad**  
[GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)
