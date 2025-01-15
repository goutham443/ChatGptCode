import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

st.title("Model Evaluation App")

# Initialize model variable
model = None

def get_dependent_variable(data):
    """Prompt user to select the dependent variable."""
    return st.selectbox("Select the dependent variable (target column):", data.columns.tolist())

# Function to preprocess data
def preprocess_data(data, dependent_variable):
    """Convert categorical columns to numeric and handle preprocessing."""
    # Handle missing values
    data = data.fillna(data.mean(numeric_only=True))  # Impute numeric NaN values with column mean
    data = data.fillna("missing")  # Replace non-numeric NaN with a placeholder

    X = data.drop([dependent_variable], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
    y = data[dependent_variable]
    return X, y

# Upload training file
st.header("Upload Training Data")
train_file = st.file_uploader("Upload your training dataset (CSV file)", type="csv", key="train")

dependent_variable = None

if train_file is not None:
    train_data = pd.read_csv(train_file)
    st.write("Training Data Preview:", train_data.head())

    # Display column names for debugging
    st.write("Columns in Training Data:", train_data.columns.tolist())

    # Allow user to select the dependent variable
    dependent_variable = get_dependent_variable(train_data)

    if dependent_variable:
        # Validate that the dependent variable is numeric
        if not np.issubdtype(train_data[dependent_variable].dtype, np.number):
            st.error(f"The selected dependent variable '{dependent_variable}' is not numeric. Please select a numeric column for regression.")
        else:
            # Preprocess training data
            X_train, y_train = preprocess_data(train_data, dependent_variable)

            # Fit the regression model
            model = LinearRegression().fit(X_train, y_train)
            
            # Model evaluation on training data
            train_r_squared = model.score(X_train, y_train)
            train_predictions = model.predict(X_train)
            train_rmse = calculate_rmse(y_train, train_predictions)

            st.subheader("Training Data Metrics")
            st.write(f"R-squared: {train_r_squared:.4f}")
            st.write(f"RMSE: {train_rmse:.4f}")

# Upload test file
st.header("Upload Test Data")
test_file = st.file_uploader("Upload your test dataset (CSV file)", type="csv", key="test")

if test_file is not None:
    test_data = pd.read_csv(test_file)
    st.write("Test Data Preview:", test_data.head())

    # Display column names for debugging
    st.write("Columns in Test Data:", test_data.columns.tolist())

    if model is not None and dependent_variable is not None:
        # Ensure the dependent variable exists in the test data
        if dependent_variable in test_data.columns:
            # Validate that the dependent variable is numeric in test data
            if not np.issubdtype(test_data[dependent_variable].dtype, np.number):
                st.error(f"The dependent variable '{dependent_variable}' in the test data is not numeric. Please ensure it matches the format in the training data.")
            else:
                # Preprocess test data
                X_test, y_test = preprocess_data(test_data, dependent_variable)

                # Predict on test data
                test_predictions = model.predict(X_test)
                test_rmse = calculate_rmse(y_test, test_predictions)

                st.subheader("Test Data Metrics")
                st.write(f"RMSE: {test_rmse:.4f}")
        else:
            st.error(f"The column '{dependent_variable}' was not found in the test data. Please ensure your dataset contains this column.")
    else:
        st.error("Please upload and train on the training data first before testing.")
