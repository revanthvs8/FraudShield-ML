import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="FraudShield ML",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent

API_URL = "http://127.0.0.1:8000"

# ---------------------------------------
# Custom Styling
# ---------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------
# Header
# ---------------------------------------

st.title("🛡️ FraudShield ML")
st.subheader("Intelligent Financial Fraud Detection")

st.write(
    "Analyze transaction features and estimate the probability "
    "that a transaction may be fraudulent."
)

st.divider()

# ---------------------------------------
# Sidebar
# ---------------------------------------

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Transaction Prediction",
        "Fraud Analytics",
        "Model Performance",
        "About",
    ],
)

# ---------------------------------------
# API Status
# ---------------------------------------

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        if response.ok:
            return True

    except requests.RequestException:
        pass

    return False


api_available = check_api()

if api_available:
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API Disconnected")

# ---------------------------------------
# Dashboard Page
# ---------------------------------------

if page == "Dashboard":

    st.header("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Machine Learning Model",
        "Logistic Regression",
    )

    col2.metric(
        "Input Features",
        "30",
    )

    col3.metric(
        "API Status",
        "Online" if api_available else "Offline",
    )

    st.info(
        "Use Transaction Prediction to estimate risk for "
        "a single transaction."
    )

    st.write("### Available Features")

    st.markdown(
        """
        - Transaction-level fraud risk prediction
        - Fraud probability and risk score
        - Transaction dataset analytics
        - Model performance reporting
        """
    )

# ---------------------------------------
# Transaction Prediction Page
# ---------------------------------------

elif page == "Transaction Prediction":

    st.header("Transaction Risk Analysis")

    st.warning(
        "This model expects the Time, V1–V28, and Amount features "
        "used by the original anonymized research dataset. "
        "It is a demonstration, not a production banking system."
    )

    with st.form("transaction_form"):

        transaction_time = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=50000.0,
        )

        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=250.0,
        )

        st.write("### Transaction Features")

        feature_values = {}

        cols = st.columns(4)

        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                feature_values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.4f",
                    key=f"feature_{i}",
                )

        submit = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True,
        )

    if submit:

        transaction = {
            "Time": transaction_time,
            **feature_values,
            "Amount": transaction_amount,
        }

        if not api_available:
            st.error(
                "The prediction API is unavailable. "
                "Start the FastAPI server and try again."
            )

        else:
            try:
                with st.spinner("Analyzing transaction..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=transaction,
                        timeout=15,
                    )

                if response.ok:

                    result = response.json()

                    probability = (
                        result["fraud_probability"] * 100
                    )

                    st.divider()

                    st.subheader("Prediction Result")

                    left, right = st.columns(2)

                    left.metric(
                        "Predicted Status",
                        result["prediction"],
                    )

                    right.metric(
                        "Fraud Risk Score",
                        f"{result['risk_score']:.2f}/100",
                    )

                    st.write("### Estimated Fraud Probability")

                    st.progress(
                        min(
                            max(
                                result["fraud_probability"],
                                0.0,
                            ),
                            1.0,
                        )
                    )

                    st.write(
                        f"{probability:.2f}% estimated probability"
                    )

                    if result["prediction"] == "Potentially Fraudulent":
                        st.error(
                            "The model flagged this transaction "
                            "as potentially fraudulent."
                        )
                    else:
                        st.success(
                            "The model classified this transaction "
                            "as likely legitimate."
                        )

                    st.caption(result["disclaimer"])

                else:
                    st.error(
                        f"API returned HTTP {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.RequestException as error:
                st.error(f"Could not contact API: {error}")

# ---------------------------------------
# Fraud Analytics Page
# ---------------------------------------

elif page == "Fraud Analytics":

    st.header("Fraud Analytics")

    data_path = PROJECT_ROOT / "data" / "creditcard.csv"

    if not data_path.exists():

        st.warning(
            "Dataset not found. Place creditcard.csv in the data folder."
        )

    else:

        df = pd.read_csv(data_path)

        total_transactions = len(df)

        fraudulent = int(
            (df["Class"] == 1).sum()
        )

        legitimate = int(
            (df["Class"] == 0).sum()
        )

        fraud_rate = (
            fraudulent / total_transactions * 100
            if total_transactions
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}",
        )

        col2.metric(
            "Fraudulent",
            f"{fraudulent:,}",
        )

        col3.metric(
            "Legitimate",
            f"{legitimate:,}",
        )

        col4.metric(
            "Fraud Rate",
            f"{fraud_rate:.3f}%",
        )

        st.write("### Transaction Class Distribution")

        class_chart = pd.DataFrame(
            {
                "Transaction Type": [
                    "Legitimate",
                    "Fraudulent",
                ],
                "Count": [
                    legitimate,
                    fraudulent,
                ],
            }
        ).set_index("Transaction Type")

        st.bar_chart(class_chart)

        st.write("### Transaction Amount Distribution")

        st.bar_chart(
            df.groupby("Class")["Amount"].mean().rename(
                index={
                    0: "Legitimate",
                    1: "Fraudulent",
                }
            )
        )

        st.caption(
            "These charts describe the research dataset. "
            "They do not represent live banking transactions."
        )

# ---------------------------------------
# Model Performance Page
# ---------------------------------------

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Review the evaluation results of the FraudShield ML model."
    )

    report_path = (
        PROJECT_ROOT / "reports" / "model_evaluation.txt"
    )

    if report_path.exists():

        with open(report_path, "r", encoding="utf-8") as file:
            report_text = file.read()

        st.subheader("Evaluation Report")

        st.code(report_text, language="text")

        st.download_button(
            label="Download Evaluation Report",
            data=report_text,
            file_name="model_evaluation.txt",
            mime="text/plain",
        )

    else:

        st.warning(
            "Evaluation report not found. "
            "Run the following command in your terminal:"
        )

        st.code("python src/train_model.py", language="bash")

    st.info(
        "This is an educational fraud detection project. "
        "Model predictions are estimates and should not be used "
        "as the sole basis for financial decisions."
    )


# ---------------------------------------
# About Page
# ---------------------------------------

elif page == "About":

    st.header("About FraudShield ML")

    st.write(
        """
        FraudShield ML is a machine learning project that estimates
        the risk of financial transaction fraud.

        The application combines a trained Logistic Regression model,
        a FastAPI prediction service, and a Streamlit dashboard.
        """
    )

    st.write("### Technology Stack")

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - FastAPI
        - Streamlit
        - Git and GitHub
        """
    )

    st.write("### Project Disclaimer")

    st.info(
        "This is an educational fraud detection prototype. "
        "Predictions are estimates and should not be used alone "
        "to make real financial decisions."
    )
