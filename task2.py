import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to load the dataset
def load_data(url):
    data = pd.read_csv(url)
    return data

# Function to preprocess the data
def preprocess_data(data):
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Function to determine the optimal number of clusters using the elbow method
def optimal_clusters(X_scaled):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot the elbow method
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Function to fit KMeans and return the cluster labels
def fit_kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(X_scaled)
    return y_kmeans, kmeans

# Function to visualize the clusters
def visualize_clusters(X_scaled, y_kmeans, kmeans):
    plt.figure(figsize=(10, 5))
    for i in range(kmeans.n_clusters):
        plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], s=100, label=f'Cluster {i + 1}')

    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.legend()
    plt.show()

# Main function to run the clustering analysis
def main(url):
    data = load_data(url)
    print(data.head())  # Display the first few rows of the dataset
    
    X_scaled = preprocess_data(data)
    optimal_clusters(X_scaled)  # Determine the optimal number of clusters
    
    # Fit the K-means model with the optimal number of clusters (e.g., 5)
    optimal_clusters_count = 5
    y_kmeans, kmeans = fit_kmeans(X_scaled, optimal_clusters_count)
    
    # Visualize the clusters
    visualize_clusters(X_scaled, y_kmeans, kmeans)

# URL to the dataset
url = 'add your dataset here'
main(url)
