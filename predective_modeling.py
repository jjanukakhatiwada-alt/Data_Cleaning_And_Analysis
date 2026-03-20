import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Excel
df = pd.read_excel("MobileSalesData.xlsx")

# Create 'Date' column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Example: Create churn column (simulate)
# Here we assume if Customer hasn't bought in last 90 days, they churned
last_purchase = df.groupby('Customer Name')['Date'].max().reset_index()
import datetime
today = df['Date'].max()
last_purchase['Days_Since_Last'] = (today - last_purchase['Date']).dt.days
last_purchase['Churn'] = last_purchase['Days_Since_Last'].apply(lambda x: 1 if x>90 else 0)

# Merge churn back into main dataset (use last record per customer)
df = df.merge(last_purchase[['Customer Name','Churn']], on='Customer Name', how='left')


# Select features
features = ['Units Sold', 'Price Per Unit', 'Customer Age', 'Brand', 'City', 'Payment Method', 'Customer Ratings']

# Encode categorical features
for col in ['Brand','City','Payment Method']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
df['Churn_Prediction'] = clf.predict(X)
df.to_csv("Customer_Churn_Predictions.csv", index=False)