# Titanic Survival Predictor

A supervised-learning portfolio project that explores passenger survival classification and exposes predictions through a Streamlit interface.

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results and Insights](#-results-and-insights)
- [Screenshots](#-screenshots)
- [Challenges and Solutions](#-challenges-and-solutions)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)
- [Author](#-author)
- [License](#-license)

## 📖 Overview

A supervised-learning portfolio project that explores passenger survival classification and exposes predictions through a Streamlit interface.

## 🎯 Problem Statement

The project frames an analytical or engineering question using the data, code, and outputs available in this repository. Findings should be interpreted according to the documented dataset, methodology, and limitations.

## 🎯 Objectives

- Understand the data and project question.
- Document the implemented analytical or engineering workflow.
- Report measurable results only where supported by repository evidence.
- Make the work reproducible for reviewers.

## ✨ Features

The features below are limited to the work described in the existing repository documentation.

### Modeling workflow

1. Inspect the training and test schemas.
2. Explore target balance and missing data.
3. Drop identifier/text fields selected by the notebook and impute missing Age values using the training median.
4. Encode categorical features and train the implemented classifier.
5. Save the model and expose inference through Streamlit.
6. Validate that the app uses the same feature order and preprocessing assumptions as training.

## 📂 Dataset

### Dataset metrics

Saved notebook output and the repository CSV files show:

| Metric | Training file | Test file |
|---|---:|---:|
| Rows | 891 | 418 |
| Columns | 12 | 11 |
| Target field | `Survived` | Not present |

In the training data, 342 passengers survived and 549 did not, giving a training-set survival proportion of **38.38%**. The training notebook reports missing values in Age (177), Cabin (687), and Embarked (2); the other listed fields have no missing values in the saved output.

## 🛠️ Tech Stack

See the imports, dependency files, SQL scripts, and application code in the repository. Only technologies directly evidenced in those files should be listed as used.

## 🔄 Project Workflow

### Modeling workflow

1. Inspect the training and test schemas.
2. Explore target balance and missing data.
3. Drop identifier/text fields selected by the notebook and impute missing Age values using the training median.
4. Encode categorical features and train the implemented classifier.
5. Save the model and expose inference through Streamlit.
6. Validate that the app uses the same feature order and preprocessing assumptions as training.

## 🗂️ Repository Structure

### Project files

| File | Purpose |
|---|---|
| `Titanic_train.csv` | Labeled training dataset |
| `Titanic_test.csv` | Test records without survival labels |
| `Titanic Survival Predictor.ipynb` | EDA, preprocessing, and model training |
| `app.py` | Streamlit application |
| `model.pkl` | Saved model artifact |
| `requirements.txt` | Python dependencies |
| `APP_LINK.md` | Deployment information, if current |

## ⚙️ Installation

### Prerequisites\n\nUse the runtime and dependencies required by the actual source files. Check the repository for a requirements or environment file before installing packages.\n\n### Setup\n\n\`\`\`bash\ngit clone https://github.com/sahil2003-ai/${repo}.git\ncd ${repo}\n\`\`\`

## ▶️ Usage

### Run locally

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

## 📊 Results and Insights

No numerical results were independently confirmed in the current README content. Add metrics only after reproducing the calculations from the repository data or saved outputs.

## Additional Project Documentation\n\n### Reported training-set performance

The saved notebook reports the following **in-sample training metrics**:

| Metric | Result |
|---|---:|
| Accuracy | 80.02% |
| Precision | 76.11% |
| Recall | 69.88% |
| F1-score | 72.87% |

**Important:** These are measurements on the same 891 rows used to fit the model, not a held-out validation or test score. They can overstate generalization and should not be presented as production performance. The 418-row test CSV has no target labels, so it cannot independently provide accuracy. A defensible next step is a stratified train/validation split or cross-validation on the labeled training data, followed by a single final refit if required.

The model workflow also calculates a ROC curve using training predictions; that AUC is not a held-out estimate and is not promoted here as such.

### Author

Sahil Gaikwad · [GitHub](https://github.com/sahil2003-ai) · [LinkedIn](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)\n## 🖼️ Screenshots

No verified screenshot path was identified in the existing README. Add real screenshots from the repository only when the corresponding image files exist.

## 🧩 Challenges and Solutions

### Validation and reproducibility

| Challenge | Approach |
|---|---|
| Metrics can be mistaken for independently verified results | Preserve the evidence status stated in the existing documentation. |
| Data sources, filters, or execution environments may differ | Confirm source paths, schema, dependencies, and calculation definitions when rerunning. |
| Screenshots and dashboard outputs may not be independently accessible | Only link to real files present in the repository. |

## ⚠️ Limitations

### Limitations

- Training metrics are in-sample only.
- Passenger survival is a historical classification exercise, not a causal analysis.
- Dataset source and license should be documented.
- The saved model and application should be tested together after any feature/preprocessing changes.

## 🚀 Future Improvements

- [ ] Document dataset source, license, and data dictionary.\n- [ ] Add reproducible execution instructions and dependency versions.\n- [ ] Export validated result tables and genuine screenshots.\n- [ ] Add tests or cross-tool checks for important calculations.

## 📚 What I Learned

### Skills demonstrated

Python, pandas, exploratory data analysis, missing-value handling, categorical encoding, logistic/classification modeling as implemented, evaluation metrics, model serialization, and Streamlit.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

No verified license statement was found in the README. Check the repository's LICENSE file and the source dataset's reuse terms before redistribution.
