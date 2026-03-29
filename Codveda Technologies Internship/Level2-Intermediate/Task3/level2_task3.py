import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. LOAD THE DATA
df = pd.read_csv('churn-bigml-80.csv')

# 2. PREPROCESS: Select only numeric columns for clustering
# We exclude non-numeric columns like 'State' and 'International plan'
X = df.select_dtypes(include=['float64', 'int64'])

# 3. FEATURE SCALING (Objective 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. ELBOW METHOD: Finding the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow (Objective 2)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker='o', color='purple')
plt.title('Objective 2: The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

# 5. K-MEANS CLUSTERING
# Based on common results for this data, we use k=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# 6. PCA: Visualizing in 2D Space (Objective 3)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Objective 3: Customer Clusters (2D PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

# 7. INTERPRETATION (Objective 4)
print("--- Cluster Summary (Average Usage) ---")
print(df.groupby('Cluster')[['Total day minutes', 'Total eve minutes', 'Customer service calls']].mean())