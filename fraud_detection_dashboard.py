import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load model and data
model = joblib.load('Fraud_detection_dtree.pkl')
data = pd.read_csv('dashboard_dataset.csv')

# Ensure feature columns are aligned with the training set
expected_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# Fill missing columns with zeros if necessary
for col in expected_columns:
    if col not in data.columns:
        data[col] = 0

# Add title
st.title("Online Payments Fraud Detection Dashboard")

# User input section for predicting fraud
st.header("Transaction Input for Fraud Detection")
with st.form(key='transaction_form'):
    amount = st.number_input("Transaction Amount", min_value=0.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)

    # Transaction type selection
    transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    
    # One-hot encoding the transaction type
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

    submit_button = st.form_submit_button(label='Predict Fraud')

# Prediction logic
if submit_button:
    input_data = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show prediction result
    if prediction == 1:
        st.error("Warning: This transaction is predicted to be fraudulent!")
    else:
        st.success("This transaction is predicted to be safe.")

# Visualization of past transactions (optional)
st.header("Transaction Insights")
option = st.selectbox("Choose a visualization:", 
                      ['Transaction Amounts', 'Model Performance'])

if option == 'Transaction Amounts':
    st.subheader("Transaction Amount Distribution")
    fig_fraud = px.histogram(data[data['isFraud'] == 1], x="amount", title="Fraud Transactions")
    st.plotly_chart(fig_fraud)

    fig_non_fraud = px.histogram(data[data['isFraud'] == 0], x="amount", title="Non-Fraud Transactions")
    st.plotly_chart(fig_non_fraud)

elif option == 'Model Performance':
    st.subheader("Model Performance Metrics")
    if 'isFraud' in data.columns:
        y_true = data['isFraud']
        y_pred = model.predict(data[expected_columns])

        report = classification_report(y_true, y_pred, output_dict=True)
        st.json(report)

        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
    else:
        st.warning("Actual 'isFraud' labels are not available for model performance evaluation.")