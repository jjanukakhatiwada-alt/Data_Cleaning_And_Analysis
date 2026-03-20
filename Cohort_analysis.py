import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Prepare Data
df = pd.read_excel("MobileSalesData.xlsx")
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
df['Revenue'] = df['Units Sold'] * df['Price Per Unit']

# 2. Assign Cohorts
# The month the customer made their very first purchase
df['OrderMonth'] = df['Date'].dt.to_period('M')
df['CohortMonth'] = df.groupby('Customer Name')['Date'].transform('min').dt.to_period('M')

# 3. Calculate Cohort Index (Months since first purchase)
def get_date_int(df, column):
    year = df[column].astype('datetime64[ns]').dt.year
    month = df[column].astype('datetime64[ns]').dt.month
    return year, month

order_year, order_month = get_date_int(df, 'OrderMonth')
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')

years_diff = order_year - cohort_year
months_diff = order_month - cohort_month

# +1 so the first month is index 1
df['CohortIndex'] = years_diff * 12 + months_diff + 1

# 4. Aggregate Data for Heatmap
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer Name'].nunique().reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Customer Name')

# 5. Calculate Retention Rates (%)
# Divide every column by the first column (the total size of that cohort)
cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0)

# 6. Visualize the Cohort Analysis
plt.figure(figsize=(12, 8))
plt.title('Customer Retention Rates (%)', fontsize=16)
sns.heatmap(data=retention, 
            annot=True, 
            fmt='.0%', 
            vmin=0.0, 
            vmax=0.5, # Adjust based on your data spread
            cmap='YlGnBu')

plt.ylabel('Cohort (First Purchase Month)')
plt.xlabel('Months Since Joining')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup (Same as before)
df = pd.read_excel("MobileSalesData.xlsx")
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
df['Revenue'] = df['Units Sold'] * df['Price Per Unit']
df['OrderMonth'] = df['Date'].dt.to_period('M')
df['CohortMonth'] = df.groupby('Customer Name')['Date'].transform('min').dt.to_period('M')

# Calculate Cohort Index
def get_date_int(df, column):
    year = df[column].astype('datetime64[ns]').dt.year
    month = df[column].astype('datetime64[ns]').dt.month
    return year, month

order_year, order_month = get_date_int(df, 'OrderMonth')
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
df['CohortIndex'] = (order_year - cohort_year) * 12 + (order_month - cohort_month) + 1

# 2. CALCULATE TOTAL REVENUE PER COHORT
cohort_revenue = df.groupby(['CohortMonth', 'CohortIndex'])['Revenue'].sum().reset_index()
revenue_pivot = cohort_revenue.pivot(index='CohortMonth', columns='CohortIndex', values='Revenue')

# 3. CALCULATE ARPU (Average Revenue Per User)
# We divide the total revenue by the total number of unique customers in the START of that cohort
cohort_sizes = df[df['CohortIndex'] == 1].groupby('CohortMonth')['Customer Name'].nunique()
arpu_pivot = revenue_pivot.divide(cohort_sizes, axis=0)

# 4. Visualize ARPU (This is usually the most insightful)
plt.figure(figsize=(14, 10))
sns.heatmap(arpu_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Average Revenue Per Customer (ARPU) by Cohort', fontsize=16)
plt.ylabel('Cohort (First Purchase Month)')
plt.xlabel('Months Since Joining')
plt.show()