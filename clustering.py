import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from matplotlib import colors

pd.set_option('display.max_columns', None)
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]

#Loading the dataset
data = pd.read_csv("./data/marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
print(data.head())

#To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

# Convert date from string to actual date time
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
data['Dt_Customer'] = (data['Dt_Customer'] - data['Dt_Customer'].min()).dt.days

#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables in the dataset:", object_cols)

#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)

#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
print(scaled_ds.head())
print(" ")

#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
print(PCA_ds.describe().T)

#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
#fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(x,y,z, c="maroon", marker="o" )
#ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
#plt.show()

# 3D Scatter plot for clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# Quick examination of elbow method to find numbers of clusters to make.
#print('Elbow Method to determine the number of clusters to be formed:')
#Elbow_M = KElbowVisualizer(KMeans(), k=10)
#Elbow_M.fit(PCA_ds)
#Elbow_M.show()

#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data["Clusters"]= yhat_AC


# 3D Scatter plot for clusters
#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection="3d")

# Plot each cluster with a different color
#scatter = ax.scatter(x, y, z, c=PCA_ds["Clusters"], cmap=cmap, s=50)
#ax.set_title("3D Projection of Clusters using PCA")
#ax.set_xlabel("Principal Component 1")
#ax.set_ylabel("Principal Component 2")
#ax.set_zlabel("Principal Component 3")

# Adding a color bar for the clusters
#legend = ax.legend(*scatter.legend_elements(), title="Clusters")
#ax.add_artist(legend)

#plt.show()

#Plotting the clusters
#fig = plt.figure(figsize=(10,8))
#ax = plt.subplot(111, projection='3d', label="bla")
#ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap )
#ax.set_title("The Plot Of The Clusters")
#plt.show()

#pl = sns.countplot(x=data["Clusters"], palette= pallet)
#pl.set_title("Distribution Of The Clusters")
#plt.show()

#pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["Clusters"], palette= pallet)
#pl.set_title("Cluster's Profile Based On Income And Spending")
##plt.legend()
#plt.show()

#plt.figure()
#pl=sns.swarmplot(x=data["Clusters"], y=data["Spent"], color= "#CBEDDD", alpha=0.5 )
#pl=sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pallet)
#plt.show()

#Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
data["Children"]=data["Kidhome"]+data["Teenhome"]
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)
#Plotting count of total campaign accepted.
#plt.figure()
#pl = sns.countplot(x=data["Total_Promos"],hue=data["Clusters"], palette= pallet)
#pl.set_title("Count Of Promotion Accepted")
#pl.set_xlabel("Number Of Total Accepted Promotions")
#plt.show()

#Plotting the number of deals purchased
#plt.figure()
#pl=sns.boxenplot(y=data["NumDealsPurchases"],x=data["Clusters"], palette= pallet)
#pl.set_title("Number of Deals Purchased")
#plt.show()



#Personal = [ "Kidhome","Teenhome","Dt_Customer", "Year_Birth", "Children", "Family_Size", "Is_Parent", "Education"]
Personal = [ "Income"]

for i in Personal:
    plt.figure()
    g= sns.jointplot(x=data[i], y=data["Spent"], hue =data["Clusters"], palette=pallet)
    # Set x-axis limits
    g.ax_joint.set_xlim(0, 120000)  # Change these values based on your dataset and desired range

    plt.show()

# 2D Scatter plot for clusters using the first two PCA components
#plt.figure(figsize=(10, 8))
#sns.scatterplot(x=PCA_ds["col1"], y=PCA_ds["col2"], hue=PCA_ds["Clusters"], palette=pallet, s=100)
#plt.title("2D Projection of Clusters using PCA")
#plt.xlabel("Principal Component 1")
#plt.ylabel("Principal Component 2")
#plt.show()