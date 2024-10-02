# Import necessary libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
# Fields: age,sex,bmi,children,smoker,region,charges
data = pd.read_csv('./data/insurance.csv')
data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

X = data_encoded.drop(columns=['charges'])
y = data_encoded['charges']
print(X.columns)

# Step 1: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA (specifying the number of components, e.g., 2 components)
pca = PCA(n_components=2)  # You can adjust the number of components based on your requirement
X_pca = pca.fit_transform(X_scaled)

pca_components = pca.components_
loadings_df = pd.DataFrame(pca_components, columns=X.columns)

plt.figure(figsize=(10, 8))
sns.heatmap(loadings_df.T, cmap='coolwarm', annot=True)
plt.title('PCA Loadings (Original Features to Components)')
plt.xlabel('Principal Components')
plt.ylabel('Original Features')
plt.show()

# Step 3: Convert the PCA results into a DataFrame (optional)
#pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])

# Step 4: Add the target variable back to the PCA DataFrame for analysis
#pca_df['charges'] = data['charges'].values

# Step 5: Visualize the variance explained by each principal component (optional)
#explained_variance = pca.explained_variance_ratio_
#print(f'Explained variance by components: {explained_variance}')

# Step 6: Visualize the PCA results (optional)
#plt.figure(figsize=(8, 6))
#plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=pca_df['charges'], cmap='viridis')
#plt.colorbar(label='charges')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.title('PCA of Insurance Charges Data')
#plt.show()

