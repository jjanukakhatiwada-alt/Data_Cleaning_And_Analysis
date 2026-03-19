import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the full dataset
data = pd.read_csv("output/full_sales_data.csv")
data.columns = data.columns.str.strip()

# Extract date features (if not already)
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Day_Name'] = data['Date'].dt.day_name()

# Target variable
y = data['Total Sales']

# Features to use
X = data[['Day', 'Month', 'Year', 'Day_Name', 'Brand', 'Customer_Age_Group', 
          'City', 'Payment Method', 'Mobile Model']]

# Identify categorical and numeric columns
categorical_cols = ['Day_Name', 'Brand', 'Customer_Age_Group', 'City', 'Payment Method', 'Mobile Model']
numeric_cols = ['Day', 'Month', 'Year']

# Column transformer to encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Transform features
X_transformed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Build regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Coefficients
feature_names = preprocessor.get_feature_names_out(X.columns)
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})

# Aggregate coefficients by variable to find most associated variables
coef_df['Variable'] = coef_df['Feature'].apply(lambda x: x.split('__')[1] if '__' in x else x)
assoc_strength = coef_df.groupby('Variable')['Coefficient'].apply(lambda x: np.sum(np.abs(x))).sort_values(ascending=False)

print("\nTop contributors (variables) to Total Sales:")
print(assoc_strength)

# Optional: display full coefficient table
print("\nAll feature coefficients:")
print(coef_df.sort_values(by='Coefficient', ascending=False))


# Save regression results
coef_df.to_csv("output/coefficients.csv", index=False)

# Save correlation results
correlations.to_csv("output/correlations.csv")

# Save aggregated tables
daily_sales.to_csv("output/daily_sales.csv", index=False)