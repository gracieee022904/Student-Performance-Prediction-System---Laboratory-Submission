# Student Performance Prediction System

A classroom prototype that estimates whether a student is **at risk of failing the final Mathematics grade**, built for the Data Mining Laboratory.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of students who may need additional academic support, using historical student records.

## Dataset
[UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Mathematics records, Portuguese secondary schools). Downloaded automatically by `train_model.py` and saved as `student_data.csv`.

## Prediction target
`G3` (final grade): **G3 < 10 = AT RISK**, **G3 >= 10 = NOT AT RISK**.
`G3` is the target only and is never used as an input feature (to avoid target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest pipeline), Streamlit, ucimlrepo, joblib.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, prints metrics, saves student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it, review the predictions, and download the results.

## Model evaluation
`train_model.py` prints accuracy plus precision, recall and F1-score (At-risk class) and saves them to `model_metrics.json`. Fill in your run's results:

| Metric | Value |
| --- | --- |
| Accuracy | |
| Precision (At risk) | |
| Recall (At risk) | |
| F1-score (At risk) | |

## Files
- `train_model.py` - dataset retrieval, preprocessing, training, evaluation
- `app.py` - Streamlit dashboard with CSV upload, column validation, prediction, and export
- `requirements.txt` - dependencies

## Limitations and responsible use
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- Small dataset (~395 records), so results vary with the train/test split.
- G1 and G2 (earlier period grades) are strong predictors, so the model is most useful later in the term.
- Predictions are not diagnoses and must not be the sole basis for real student decisions. An "AT RISK" flag should prompt a supportive conversation and review by a human.
