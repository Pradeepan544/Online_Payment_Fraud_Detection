# 💳 Online Payment Fraud Detection
[Try it out here!](https://onlinepaymentsfrauddetection544.streamlit.app/)

## 📚 Overview
This repository contains a machine learning project aimed at detecting fraudulent online payment transactions. The dataset, sourced from Kaggle user Rupak Roy, has been thoroughly analyzed and processed to build an effective fraud detection model.

## 📁 Contents
- **Dataset**: Raw data used for training and testing the model.
- **Exploratory Data Analysis (EDA)**: Insights gained from analyzing the dataset.
- **Model Development**: Implementation of various machine learning algorithms.
- **Deployment**: Link to the deployed application.

## 📊 Dataset
The dataset was obtained from Kaggle, specifically from the user Rupak Roy. You can download the dataset [here](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/data).

It includes various features related to online payment transactions, focusing on identifying fraudulent activities. The target variable indicates whether a transaction is fraudulent.

## 🔍 Data Preprocessing
The dataset exhibited a significant class imbalance with a ratio of 99.9% non-fraudulent transactions to 0.08% fraudulent transactions. Preprocessing steps included:
- Handling missing values.
- Resampling techniques to address class imbalance, ensuring the model learns effectively from both classes.

## 📝 Model Development
Implemented several machine learning algorithms, including:
- **Decision Tree**
- **K-Nearest Neighbour**
- **Support Vector Machine**
- **Logistic Regression**

These models were trained and evaluated to determine their effectiveness in detecting fraudulent transactions. Feature importance analysis was also performed to identify which factors are most significant in predicting fraud.

## 🖥️ Deployment
The dashboard can be accessed [here](https://onlinepaymentsfrauddetection544.streamlit.app/) where users can interact with the model and visualize predictions.

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments
Special thanks to Kaggle and Rupak Roy for sharing the dataset for fraud detection.
