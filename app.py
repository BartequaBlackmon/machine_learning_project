import pandas as pd
import streamlit as st 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from ipykernel import kernelapp as app

# Upload dataset section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset successfully uploaded!")
else:
    # Load default data
    date_df = pd.read_csv("dateinfo.csv")
    sell_df = pd.read_csv("selldata.csv")
    transaction_df = pd.read_csv("transactiondata.csv")

    # Data preprocessing
    transaction_df['CALENDAR_DATE'] = pd.to_datetime(transaction_df['CALENDAR_DATE'], format='%m/%d/%y')
    date_df['CALENDAR_DATE'] = pd.to_datetime(date_df['CALENDAR_DATE'], format='%m/%d/%y')

    df = sell_df.merge(transaction_df, on='SELL_ID', how='left')

# Sidebar filter
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

# Summary Statistics Section
st.subheader("Summary Statistics")

if not filtered_df.empty:
    # Calculating metrics
    total_sales = (filtered_df['QUANTITY'] * filtered_df['PRICE']).sum()  # Fixed total sales logic
    avg_sales = filtered_df['QUANTITY'].mean()
    top_date = filtered_df.loc[filtered_df['QUANTITY'].idxmax(), 'CALENDAR_DATE']
    top_quantity = filtered_df['QUANTITY'].max()

    # Display metrics
    st.markdown(f"""
    - **Total Sales Revenue**: ${total_sales:,.2f}
    - **Average Daily Sales (Quantity)**: {avg_sales:.2f} units
    - **Top Performing Day**: {top_date.date()} ({top_quantity:.0f} units sold)
    """)
else:
    st.write("No data available for the selected filters.")


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

# Additional Visualization: Heatmap of Sales by Day
st.subheader("Sales Heatmap")
if not filtered_df.empty:
    heatmap_data = filtered_df.groupby(filtered_df['CALENDAR_DATE']).sum().reset_index()
    heatmap_data['CALENDAR_DATE'] = pd.to_datetime(heatmap_data['CALENDAR_DATE']).dt.strftime('%d/%m/%y')
    heatmap_data['Day'] = pd.to_datetime(heatmap_data['CALENDAR_DATE']).dt.strftime('%A')
    pivot_data = heatmap_data.pivot_table(values='QUANTITY', index='Day', columns='CALENDAR_DATE', fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_data, cmap="coolwarm", ax=ax)
    ax.set_title("Sales Distribution by Day")
    st.pyplot(fig)

# Additional Visualization: Histogram of Sales
st.subheader("Sales Distribution")
if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(filtered_df['QUANTITY'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Daily Sales")
    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Additional Visualization: Line Graph of Average Daily Sales
st.subheader("Average Daily Sales Line Graph")
if not filtered_df.empty:
    daily_avg = filtered_df.groupby('CALENDAR_DATE')['QUANTITY'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_avg['CALENDAR_DATE'], daily_avg['QUANTITY'], color='green', marker='o', label='Average Quantity Sold')
    ax.set_title("Average Daily Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Quantity Sold")
    ax.legend()
    st.pyplot(fig)

# Train Sales Forecasting Model
st.subheader("Sales Forecasting")
if st.button("Train Forecasting Model"):
    # Prepare data for regression
    filtered_df['Days'] = (filtered_df['CALENDAR_DATE'] - filtered_df['CALENDAR_DATE'].min()).dt.days
    X = filtered_df[['Days']]
    y = filtered_df['QUANTITY']
    
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
    future_dates = [(filtered_df['CALENDAR_DATE'].max() + timedelta(days=i)) for i in range(1, 8)]
    future_days = [(date - filtered_df['CALENDAR_DATE'].min()).days for date in future_dates]
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
    ax.plot(filtered_df['CALENDAR_DATE'], filtered_df['QUANTITY'], label="Actual", marker='o')
    ax.plot(forecast_df['Date'], forecast_df['Predicted Quantity'], label="Forecast", marker='o', linestyle='--')
    ax.set_title("Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.legend()
    st.pyplot(fig)


    # Scatter Plot: Actual vs Predicted
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, color='blue', edgecolor='black', alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red')
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)

else:
    st.write("Click the button to train the sales forecasting model.")

# Footer
st.write("Developed by Bartequa Blackmon with ❤️ using Streamlit.")
