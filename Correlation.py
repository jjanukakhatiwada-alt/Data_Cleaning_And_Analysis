import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import numpy as np

# Load full dataset
data = pd.read_csv("output/full_sales_data.csv")
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Day_Name'] = data['Date'].dt.day_name()

# Target and features
y = data['Total Sales']
X = data[['Day', 'Month', 'Year', 'Day_Name', 'Brand', 'Customer_Age_Group',
          'City', 'Payment Method', 'Mobile Model']]

categorical_cols = ['Day_Name', 'Brand', 'Customer_Age_Group', 'City', 'Payment Method', 'Mobile Model']
numeric_cols = ['Day', 'Month', 'Year']

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)
X_transformed = preprocessor.fit_transform(X)

# Train model
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Create coef_df
feature_names = preprocessor.get_feature_names_out(X.columns)
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})

# --- Plotting ---
coef_df['Variable'] = coef_df['Feature'].apply(lambda x: x.split('__')[1] if '__' in x else x)
assoc_strength = coef_df.groupby('Variable')['Coefficient'].apply(lambda x: np.sum(np.abs(x))).sort_values(ascending=False)

plt.figure(figsize=(10,6))
assoc_strength.plot(kind='bar', color='skyblue')
plt.ylabel("Total Contribution to Sales (absolute)")
plt.title("Most Influential Variables on Total Sales")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("output/full_sales_data.csv")
data.columns = data.columns.str.strip()  # remove any spaces

# Convert date to numeric features
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Day_Name'] = data['Date'].dt.day_name()

# Select target and independent variables
target = 'Total Sales'
independent_vars = ['Day', 'Month', 'Year', 'Day_Name', 'Brand', 'Customer_Age_Group',
                    'City', 'Payment Method', 'Mobile Model']

X = data[independent_vars]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Correlation with target
correlations = X_encoded.corrwith(data[target]).sort_values(ascending=False)

# Display top positive and negative correlations
print("Top positive correlations:\n", correlations.head(10))
print("\nTop negative correlations:\n", correlations.tail(10))


# Save correlation results
correlations.to_csv("output/correlations.csv")