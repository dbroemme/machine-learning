import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('./data/insurance.csv')

# 2. Convert categorical variables to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 3. Feature scaling for continuous variables
scaler = StandardScaler()
df_encoded[['age', 'bmi', 'children']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'children']])

# 4. Define the target and features for classification
X = df_encoded.drop(columns=['smoker_yes'])  # Features (drop the target)
y = df_encoded['smoker_yes']  # Target (categorical)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize the SVM model
model = SVC(kernel='rbf', probability=True)  # Using the Radial Basis Function (RBF) kernel

# 7. Train the SVM model
model.fit(X_train, y_train)

# 8. Test the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary values

# 9. Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy:.2f}")

