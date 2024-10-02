import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor

# 1. Load the data
df = pd.read_csv('./data/insurance.csv')

# 2. Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 3. Feature scaling for continuous variables
scaler = StandardScaler()
df_encoded[['age', 'bmi', 'children']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children']])

# 4. Define the target (charges) and features (drop 'charges')
X = df_encoded.drop(columns=['charges'])  # Features
y = df_encoded['charges']  # Target (continuous variable)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Use LazyRegressor to evaluate different models
regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regressor.fit(X_train, X_test, y_train, y_test)

# 7. Print the results
print(models)
