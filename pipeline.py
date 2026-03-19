import pandas as pd
from sqlalchemy import create_engine
import os

print("Pipeline started...")

# Step 1: Load Excel
df = pd.read_excel("MobileSalesData.xlsx")
print("Data loaded successfully!")
print(df.head())
print("Columns:", df.columns)

# Step 2: Clean data & create Date column
df = df.dropna()  # remove empty rows
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])  # create a Date column
df['Total Sales'] = df['Units Sold'] * df['Price Per Unit']  # calculate sales per row
print("Data cleaned successfully!")

# Step 3: Transform data
daily_sales = df.groupby("Date")["Total Sales"].sum().reset_index()
product_sales = df.groupby("Mobile Model")["Total Sales"].sum().reset_index()
region_sales = df.groupby("City")["Total Sales"].sum().reset_index()
print("Data transformed successfully!")

# Step 4: Save to database
engine = create_engine("sqlite:///sales.db")  # database in current folder
daily_sales.to_sql("daily_sales", engine, if_exists="replace", index=False)
product_sales.to_sql("product_sales", engine, if_exists="replace", index=False)
region_sales.to_sql("region_sales", engine, if_exists="replace", index=False)

# Step 5: Save CSV files to output folder
if not os.path.exists("output"):
    os.makedirs("output")  # create output folder if not exists

daily_sales.to_csv("output/daily_sales.csv", index=False)
product_sales.to_csv("output/product_sales.csv", index=False)
region_sales.to_csv("output/region_sales.csv", index=False)

print("Pipeline completed successfully!")


import pandas as pd
from sqlalchemy import create_engine
import os

print("Pipeline started...")

# Step 1: Load Excel
df = pd.read_excel("MobileSalesData.xlsx")
print("Data loaded successfully!")
print(df.head())
print("Columns:", df.columns)

# Step 2: Clean data & create Date column
df = df.dropna()  # remove empty rows
df['Date'] = pd.to_datetime(df[['Year','Month','Day']])  # create a Date column
df['Total Sales'] = df['Units Sold'] * df['Price Per Unit']  # calculate sales per row
print("Data cleaned successfully!")

# Step 3: (Optional) Group Customer Age into 5-year bins
bins = list(range(0, 101, 5))
labels = [f"{b}-{b+4}" for b in bins[:-1]]
df['Customer_Age_Group'] = pd.cut(df['Customer Age'], bins=bins, labels=labels, right=False)

# Step 4: Save aggregated tables (existing)
daily_sales = df.groupby("Date")["Total Sales"].sum().reset_index()
product_sales = df.groupby("Mobile Model")["Total Sales"].sum().reset_index()
region_sales = df.groupby("City")["Total Sales"].sum().reset_index()

# Step 5: Save full dataset for multivariate regression
full_dataset = df[['Date','Total Sales','Brand','Customer_Age_Group','City','Payment Method','Mobile Model','Day','Month','Year','Day Name']]
full_dataset.to_csv("output/full_sales_data.csv", index=False)

# Step 6: Save to database
engine = create_engine("sqlite:///sales.db")
daily_sales.to_sql("daily_sales", engine, if_exists="replace", index=False)
product_sales.to_sql("product_sales", engine, if_exists="replace", index=False)
region_sales.to_sql("region_sales", engine, if_exists="replace", index=False)
full_dataset.to_sql("full_sales_data", engine, if_exists="replace", index=False)

print("All tables saved successfully in output folder and database!")


