# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 2: Load Excel file
df = pd.read_excel("MobileSalesData.xlsx")

# Step 3: Create a 'Date' column from Day, Month, Year
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Step 4: Calculate Revenue
df['Revenue'] = df['Units Sold'] * df['Price Per Unit']

# Step 5: Loop through each year
years = df['Year'].unique()
for year in years:
    # Filter data for the year
    df_year = df[df['Year'] == year]
    
    # Aggregate daily revenue
    daily_sales = df_year.groupby('Date')['Revenue'].sum().reset_index()
    
    # Step 6: Visualize daily revenue trend for the year
    plt.figure(figsize=(12,6))
    plt.plot(daily_sales['Date'], daily_sales['Revenue'], label=f'Daily Revenue {year}')
    plt.title(f'Daily Revenue Trend for {year}')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.show()
    
    # Step 7: Prepare data for Prophet
    prophet_df = daily_sales.rename(columns={"Date": "ds", "Revenue": "y"})
    
    # Step 8: Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    # Step 9: Forecast next 30 days within the same year
    # Note: We'll forecast only until the end of the year
    last_date = daily_sales['Date'].max()
    end_of_year = pd.Timestamp(year=year, month=12, day=31)
    days_to_forecast = (end_of_year - last_date).days
    if days_to_forecast > 0:
        future = model.make_future_dataframe(periods=days_to_forecast)
        forecast = model.predict(future)
        
        # Step 10: Plot forecast
        model.plot(forecast)
        plt.title(f'Revenue Forecast for {year} (Remaining Days)')
        plt.show()
    else:
        print(f"No remaining days to forecast for {year}.")