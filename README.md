# Titanic Survival Predictor

A supervised-learning portfolio project that explores passenger survival classification and exposes predictions through a Streamlit interface.

![Status](https://img.shields.io/badge/Status-Documented-yellow)

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

The project addresses an analytical or predictive question using the data and implementation files in the repository. Results are meaningful only when dataset definitions, preprocessing, and evaluation design are transparent.

## 🎯 Objectives

- Inspect and prepare the available data.
- Implement the analysis or model workflow documented in the project.
- Report verifiable metrics and distinguish saved outputs from reproduced results.
- Make the project easier to reproduce and review.

## ✨ Features

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

1. Inspect the training and test schemas.
2. Explore target balance and missing data.
3. Drop identifier/text fields selected by the notebook and impute missing Age values using the training median.
4. Encode categorical features and train the implemented classifier.
5. Save the model and expose inference through Streamlit.
6. Validate that the app uses the same feature order and preprocessing assumptions as training.

## 🗂️ Repository Structure

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

## 🖼️ Screenshots

Add only genuine charts or application screenshots that exist in the repository. No screenshot path is asserted here unless documented in the source files.

## 🧩 Challenges and Solutions

| Challenge | Solution |
|---|---|
| Saved notebook results can become stale | Rerun cells from top to bottom and verify outputs. |
| Model performance can be overstated | Use a valid holdout or cross-validation and report the split strategy. |
| Reproduction depends on paths and package versions | Document dependencies, input paths, and data provenance. |

## ⚠️ Limitations

### Limitations

- Training metrics are in-sample only.
- Passenger survival is a historical classification exercise, not a causal analysis.
- Dataset source and license should be documented.
- The saved model and application should be tested together after any feature/preprocessing changes.

## 🚀 Future Improvements

- [ ] Document dataset source, license, and data dictionary.\n- [ ] Add reproducible execution instructions and dependency versions.\n- [ ] Export validated result tables and genuine screenshots.\n- [ ] Add tests or cross-tool checks for important calculations.

## 📚 What I Learned

- Building a documented and reproducible analysis workflow.
- Choosing metrics appropriate to the task and dataset.
- Explaining assumptions and limitations alongside results.

## 👤 Author

**Sahil Gaikwad**

- GitHub: [sahil2003-ai](https://github.com/sahil2003-ai)
- LinkedIn: [Sahil Gaikwad](https://www.linkedin.com/in/sahil-gaikwad-a76b7224a/)

## 📄 License

No verified license statement was found in the README. Check the repository's LICENSE file and the source dataset's reuse terms before redistribution.