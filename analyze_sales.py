import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load the daily sales data
daily_sales = pd.read_csv("output/daily_sales.csv")

# Ensure 'Date' column is datetime type
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

# -------------------------------
# Daily Sales Plot (your existing code)
# -------------------------------
fig_daily = px.line(daily_sales, x='Date', y='Total Sales', title='Daily Sales')
fig_daily.show()

# -------------------------------
# Aggregate Monthly Sales
# -------------------------------
# Create 'Year-Month' column
daily_sales['YearMonth'] = daily_sales['Date'].dt.to_period('M')

# Sum sales per month
monthly_sales = daily_sales.groupby('YearMonth')['Total Sales'].sum().reset_index()

# Convert 'YearMonth' back to string for plotting
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

# -------------------------------
# Monthly Sales Plot using Plotly
# -------------------------------
fig_monthly = px.bar(monthly_sales, x='YearMonth', y='Total Sales',
                     title='Monthly Sales', labels={'YearMonth':'Month', 'Total Sales':'Sales'})
fig_monthly.show()

# -------------------------------
# Optional: Monthly Sales Plot using Matplotlib
# -------------------------------
plt.figure(figsize=(10,6))
plt.bar(monthly_sales['YearMonth'], monthly_sales['Total Sales'], color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()