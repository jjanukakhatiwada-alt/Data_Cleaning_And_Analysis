import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load full dataset (IMPORTANT)
data = pd.read_csv("output/full_sales_data.csv")
data.columns = data.columns.str.strip()

# Convert date
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Day_Name'] = data['Date'].dt.day_name()

# Target
y = data['Total Sales']

# Use ALL useful features
X = data[['Day','Month','Year','Day_Name','Brand',
          'Customer_Age_Group','City','Payment Method','Mobile Model']]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("\nTop 10 Positive:")
print(coef_df.head(10))

print("\nTop 10 Negative:")
print(coef_df.tail(10))