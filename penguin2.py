import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
penguins = pd.read_csv('./data/penguins.csv')

# Select relevant numeric columns for clustering
penguins_numeric = penguins[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]

# Drop rows with missing values
penguins_numeric = penguins_numeric.dropna()

# Normalize the data
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_numeric)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(penguins_scaled)

# Add cluster labels to the original dataframe
penguins_numeric['Cluster'] = clusters

# Create scatterplot: Culmen Length vs Culmen Depth
plt.figure(figsize=(8, 6))

# Create a scatter plot with different colors for each cluster
scatter = plt.scatter(penguins_numeric['Culmen Length (mm)'], 
                      penguins_numeric['Culmen Depth (mm)'], 
                      c=penguins_numeric['Cluster'], cmap='viridis', s=100)

# Adding labels and title
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.title('Penguin Clusters (Culmen Length vs. Culmen Depth)')

# Add a legend
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, ['Cluster 0', 'Cluster 1', 'Cluster 2'], title="Penguin Cluster")

# Show the plot
plt.show()
