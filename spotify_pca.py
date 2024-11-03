import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Load the data from the CSV file
#data = pd.read_csv('../datasets/StudentPerformanceFactors.csv')
#data = data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Gender', 'Exam_Score']]

#data = pd.read_csv('./data/spotify-2023.csv', encoding="ISO-8859-1")
#data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
# Drop or fill NaN values (if there are any), e.g., filling with 0
#data['streams'].fillna(0, inplace=True)  # or df.dropna(subset=['streams'], inplace=True)
# Convert to integer
#data['streams'] = data['streams'].astype(int)
#data.to_csv("./data/modified_spotify.csv", index=False, encoding="utf-8")


data = pd.read_csv('./data/modified_spotify.csv', encoding="ISO-8859-1")
categorical_columns = ['key', 'mode', 'in_deezer_playlists', 'in_shazam_charts'] 
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[f'{column}'] = label_encoder.fit_transform(data[column])

X = data.drop(columns=['streams', 'track_name', 'artist(s)_name'])
y = data['streams']

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

