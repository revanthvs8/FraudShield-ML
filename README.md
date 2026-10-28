# 🛡️ FraudShield ML

### Machine Learning-Based Credit Card Fraud Detection

FraudShield ML is a machine learning project that estimates the risk of credit card fraud. It combines a machine learning model, a FastAPI backend, and a Streamlit dashboard to demonstrate transaction risk prediction and fraud analytics.

## Features

- Credit card fraud risk prediction
- Interactive Streamlit dashboard
- FastAPI REST API
- Fraud analytics and data visualization
- Model performance evaluation
- Transaction risk score display

## Technology Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Git and GitHub

## Project Structure

```text
FraudShield-ML/
├── app.py
├── README.md
├── requirements.txt
├── data/
├── models/
├── reports/
└── src/
    ├── api.py
    ├── train_model.py
    ├── exploratory_analysis.py
    └── visualize_data.py
```

## Dataset

This project uses a credit card transaction dataset containing anonymized numerical features.

- `Time`: Relative transaction time
- `V1`–`V28`: Anonymized transaction features
- `Amount`: Transaction amount
- `Class`: Target label; 0 represents legitimate transactions and 1 represents fraudulent transactions

The dataset is used for educational and research purposes. Check its redistribution terms before sharing it.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FraudShield-ML
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

Start the API in the first terminal:

```bash
python -m uvicorn src.api:app --reload
```

Start the dashboard in a second terminal:

```bash
streamlit run app.py
```

Open the local address displayed by Streamlit, usually `http://localhost:8501`.

## Model Evaluation

The project includes a model evaluation report. Actual precision, recall, F1-score, and other metrics should be taken from the generated evaluation results.

## Future Improvements

- Compare multiple machine learning models
- Improve fraud detection threshold selection
- Add automated tests
- Add model explainability
- Deploy the application
- Add prediction monitoring

## Disclaimer

FraudShield ML is an educational prototype. Its predictions are estimates and should not be used as the sole basis for real financial decisions.

## Author

Developed as a final-year BTech machine learning project.