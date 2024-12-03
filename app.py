import pandas as pd
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from ipykernel import kernelapp as app

date_df = pd.read_csv("dateinfo.csv")
sell_df = pd.read_csv("selldata.csv")
transaction_df = pd.read_csv("transactiondata.csv")

transaction_df['CALENDAR_DATE'] = pd.to_datetime(transaction_df['CALENDAR_DATE'], format='%m/%d/%y')
date_df['CALENDAR_DATE'] = pd.to_datetime(date_df['CALENDAR_DATE'], format='%m/%d/%y')

df = sell_df.merge(transaction_df, on='SELL_ID', how='left')

st.sidebar.header("Filter Options")
item_name = st.sidebar.selectbox("select Item", df['ITEM_NAME'].unique())
start_date = st.sidebar.date_input("Start Date", df['CALENDAR_DATE'].min())
end_date = st.sidebar.date_input("End Date", df['CALENDAR_DATE'].max())

filtered_df = df[(df['ITEM_NAME'] == item_name) &
                 (df['CALENDAR_DATE'] >= pd.to_datetime(start_date)) &
                 (df['CALENDAR_DATE'] <= pd.to_datetime(end_date))]


# Main page
st.title("Sale Analysis and Forecasting")
st.write(f"Displaying data for **{item_name}** from **{start_date}**")
st.dataframe(filtered_df)

# Visualization: Sales Trend
st.subheader("Sales Trend")
if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_df['CALENDAR_DATE'], filtered_df['QUANTITY'], marker='o', label = 'Quantity Sold')
    ax.set_title("Quantity Sold Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("No data available for the selected filters.")


# Train Sales Forecasting Model
st.subheader("Sales Forecasting")
if st.button("Train Forecasting Model"):
    # Prepare data for regression
    filtered_data['Days'] = (filtered_data['CALENDAR_DATE'] - filtered_data['CALENDAR_DATE'].min()).dt.days
    X = filtered_data[['Days']]
    y = filtered_data['QUANTITY']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    st.write(f"Model Trained! Mean Squared Error: {mse:.2f}")
    
    # Forecast future sales
    future_dates = [(filtered_data['CALENDAR_DATE'].max() + timedelta(days=i)) for i in range(1, 8)]
    future_days = [(date - filtered_data['CALENDAR_DATE'].min()).days for date in future_dates]
    future_predictions = model.predict(np.array(future_days).reshape(-1, 1))
    
    # Display forecast
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Quantity': future_predictions
    })
    st.write("Next Week's Sales Forecast:")
    st.dataframe(forecast_df)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_data['CALENDAR_DATE'], filtered_data['QUANTITY'], label="Actual", marker='o')
    ax.plot(forecast_df['Date'], forecast_df['Predicted Quantity'], label="Forecast", marker='o', linestyle='--')
    ax.set_title("Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Click the button to train the sales forecasting model.")