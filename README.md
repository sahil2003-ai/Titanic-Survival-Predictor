# Titanic Survival Predictor

> Educational classification project with a Streamlit interface for exploring predictions on the historical Titanic dataset.

![Status](https://img.shields.io/badge/Status-Portfolio--Project-blue)

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Insights](#results-and-insights)
- [Screenshots](#screenshots)
- [Challenges and Solutions](#challenges-and-solutions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [What I Learned](#what-i-learned)
- [Author](#author)
- [License](#license)

## 📖 Overview

Educational classification project with a Streamlit interface for exploring predictions on the historical Titanic dataset. This README documents the intended workflow and the files available in the repository. Verify all outputs against the current code and data before using them as evidence.

## 🎯 Problem Statement

This project demonstrates how a supervised learning pipeline can turn passenger attributes into a model prediction.

## 🎯 Objectives

- Prepare features for classification.
- Evaluate a model on held-out data.
- Expose inference through a simple interface.

## ✨ Features

- Notebook-based EDA and training.
- Saved model file and Streamlit app.
- Requirements file for dependencies.

## 📂 Dataset

The repository CSV files were inspected: `Titanic_train.csv` contains **891 rows × 12 columns**, and `Titanic_test.csv` contains **418 rows × 11 columns**.

- **Training target:** `Survived`.
- **Features include:** passenger class, sex, age, siblings/spouses, parents/children, ticket, fare, cabin, and embarkation port.
- The test file has no `Survived` column, as expected for a prediction dataset.
- Document the original dataset source and reuse terms.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Model workflow |
| pandas | Data handling |
| scikit-learn | Preprocessing and classification |
| Streamlit | Interactive app |

## 🔄 Project Workflow

1. Install dependencies from requirements.txt in a virtual environment.
2. Inspect train/test schemas and identify the target and input features.
3. Run Titanic Survival Predictor.ipynb from top to bottom.
4. Split training data before fitting imputers or encoders to avoid leakage.
5. Train the implemented classifier and compare it with a simple baseline.
6. Evaluate on held-out data using suitable classification metrics.
7. Confirm app.py and model.pkl use matching preprocessing and feature order.
8. Run streamlit run app.py and test representative inputs.

## 🗂️ Repository Structure

```text
Titanic-Survival-Predictor/
├── Titanic Survival Predictor.ipynb
├── Titanic_train.csv
├── Titanic_test.csv
├── app.py
├── model.pkl
├── requirements.txt
└── APP_LINK.md
└── README.md
```

The list above highlights known project files and folders; inspect the repository for the complete current structure.

## ⚙️ Installation

### Prerequisites

- Install the tools listed in the Tech Stack that apply to this project.
- For Python projects, use a Python version compatible with the dependencies actually imported by the code.
- For Power BI projects, use Power BI Desktop.

### Setup

git clone https://github.com/sahil2003-ai/Titanic-Survival-Predictor.git; cd Titanic-Survival-Predictor; python -m venv .venv; activate the environment; pip install -r requirements.txt

Do not run generic commands that do not match the files in this repository. Confirm the data paths, database dialect, and package versions before execution.

## ▶️ Usage

Run streamlit run app.py from the repository folder after installing dependencies.

## 📊 Results and Insights

### Verified dataset dimensions

| File | Rows | Columns |
|---|---:|---:|
| `Titanic_train.csv` | 891 | 12 |
| `Titanic_test.csv` | 418 | 11 |

The notebook contains training/EDA steps, but a reproducible held-out evaluation metric was not verified in this review. Do not claim model accuracy until the split strategy, preprocessing, and evaluation are rerun. Compare against a simple baseline and fit preprocessing only on training data.

## 🖼️ Screenshots

No image files were confirmed for this repository during this update. If you have screenshots, upload them into an `images/` folder, then embed each real path here with a short caption. Do not use placeholder image paths.

## 🧩 Challenges and Solutions

| Challenge | Solution / validation approach |
|---|---|
| Consistent preprocessing between training and app | Ensure the same fitted transformations and feature order are used for inference. |

## ⚠️ Limitations

- Dataset dimensions were verified from the repository CSV contents.
- Model accuracy was not verified from a reproducible held-out evaluation in this review.
- This is a historical educational dataset and should not be generalized to real-world survival prediction.

## 🚀 Future Improvements

- [ ] Add an app screenshot under images/.
- [ ] Document the train/test split and verified metrics.
- [ ] Add tests for feature order and input validation.

## 📚 What I Learned

- Classification pipelines and evaluation.
- Serving model predictions through Streamlit.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

A LICENSE file exists; review it for reuse terms.
