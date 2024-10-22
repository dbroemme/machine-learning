import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
penguins = pd.read_csv('./data/penguins_lter.csv')

# Select relevant numeric columns for clustering
penguins_numeric = penguins[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Species']]

# Drop rows with missing values
penguins_numeric = penguins_numeric.dropna()

# Normalize the data
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_numeric[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']])

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(penguins_scaled)

# Add cluster labels to the original dataframe
penguins_numeric['Cluster'] = clusters

# Map the clusters to the species
# Find the most common species in each cluster
species_cluster_mapping = penguins_numeric.groupby('Cluster')['Species'].agg(lambda x: x.mode()[0])

# Create scatterplot: Culmen Length vs Culmen Depth
plt.figure(figsize=(8, 6))

y_axis_field = 'Culmen Length (mm)'
#x_axis_field = 'Culmen Depth (mm)'
x_axis_field = 'Flipper Length (mm)'
# Create a scatter plot with different colors for each cluster
scatter = plt.scatter(penguins_numeric[x_axis_field], 
                      penguins_numeric[y_axis_field], 
                      c=penguins_numeric['Cluster'], cmap='viridis', s=100)

# Adding labels and title
plt.xlabel(x_axis_field)
plt.ylabel(y_axis_field)
plt.title('Penguin Clusters')

# Add a legend with species names
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
cluster_labels = [species_cluster_mapping[i] for i in range(3)]
plt.legend(handles, cluster_labels, title="Penguin Species")

# Show the plot
plt.show()




