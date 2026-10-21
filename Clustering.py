import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = sns.load_dataset("1553768847-housing.csv")

# 1. Select Features
X = df[["total_bill", "tip"]]

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find Optimal K (Elbow Method)
wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 4. Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# 5. Visualization
plt.scatter(df["total_bill"], df["tip"], c=df["cluster"])
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Clustering Result")
plt.show()