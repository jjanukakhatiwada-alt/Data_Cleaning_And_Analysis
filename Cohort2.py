import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_excel("MobileSalesData.xlsx")
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
df['Revenue'] = df['Units Sold'] * df['Price Per Unit']

# 2. Group by QUARTER instead of Month
# This reduces the "noise" in the chart
df['OrderMonth'] = df['Date'].dt.to_period('M')
df['CohortQuarter'] = df.groupby('Customer Name')['Date'].transform('min').dt.to_period('Q')

# Calculate Months Since Joining
def get_date_int(df, col):
    return df[col].astype('datetime64[ns]').dt.year, df[col].astype('datetime64[ns]').dt.month

order_year, order_month = get_date_int(df, 'OrderMonth')
# Note: Using the first month of the cohort quarter for the calculation
cohort_start_date = df['CohortQuarter'].dt.start_time
cohort_year, cohort_month = cohort_start_date.dt.year, cohort_start_date.dt.month

df['CohortIndex'] = (order_year - cohort_year) * 12 + (order_month - cohort_month) + 1

# 3. Create ARPU (Average Revenue Per User) Heatmap
# Total Revenue per Quarter-Cohort per Month
cohort_rev = df.groupby(['CohortQuarter', 'CohortIndex'])['Revenue'].sum().reset_index()
# Number of unique customers who STARTED in that Quarter
cohort_sizes = df[df['CohortIndex'] == 1].groupby('CohortQuarter')['Customer Name'].nunique()

# Pivot and divide by size
revenue_pivot = cohort_rev.pivot(index='CohortQuarter', columns='CohortIndex', values='Revenue')
arpu_pivot = revenue_pivot.divide(cohort_sizes, axis=0)

# 4. Plot (Limiting to first 12 months for extra clarity)
plt.figure(figsize=(16, 8))
sns.heatmap(arpu_pivot.iloc[:, :12], annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('Average Spend per Customer (By Quarter Joined)', fontsize=16)
plt.xlabel('Month of Relationship')
plt.ylabel('Cohort Quarter')
plt.show()