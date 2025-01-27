#Clustering on the given Datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")
products = pd.read_csv("Products.csv")

transactions = transactions.merge(products, on="ProductID")
data = transactions.merge(customers, on="CustomerID")

customer_features = (
    data.groupby("CustomerID")
    .agg({
        "TotalValue": "sum",
        "ProductID": "nunique",
        "Region": "first",
        "SignupDate": "first",
    })
    .reset_index()
)
customer_features["SignupYear"] = pd.to_datetime(customer_features["SignupDate"]).dt.year

encoder = OneHotEncoder(sparse=False)
region_encoded = encoder.fit_transform(customer_features[["Region"]])

scaler = StandardScaler()
numerical_data = scaler.fit_transform(customer_features[["TotalValue", "ProductID", "SignupYear"]])

features = np.hstack((numerical_data, region_encoded))

db_scores = []
clusters_range = range(2, 11)
for k in clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    db_scores.append(davies_bouldin_score(features, cluster_labels))

optimal_k = clusters_range[np.argmin(db_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_features["Cluster"] = kmeans.fit_predict(features)

plt.figure(figsize=(8, 5))
plt.plot(clusters_range, db_scores, marker='o')
plt.title("Davies-Bouldin Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Score")
plt.show()

sns.pairplot(customer_features, hue="Cluster", vars=["TotalValue", "ProductID", "SignupYear"])
plt.show()

customer_features.to_csv("Customer_Clusters.csv", index=False)
