import numpy as np
import matplotlib.pyplot as plt

# Sample data points
data = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0],
    [9.0, 3.0]
])

# K-means parameters
k = 3  # Number of clusters
max_iterations = 100

# Step 1: Initialize centroids randomly from the data points
centroids = data[np.random.choice(data.shape[0], k, replace=False)]
initial_centroids = centroids.copy()  # Keep a copy of initial centroids

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-means algorithm
for iteration in range(max_iterations):
    # Step 2: Assign each data point to the closest centroid
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)

    # Step 3: Update centroids to be the mean of points in each cluster
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Avoid empty clusters
            new_centroids.append(np.mean(cluster, axis=0))
        else:
            # If a cluster is empty, reinitialize a centroid randomly
            new_centroids.append(data[np.random.choice(data.shape[0])])

    new_centroids = np.array(new_centroids)

    # Check for convergence
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Epochs/iterations to convergence
epochs = iteration + 1

# Calculate the error rate (within-cluster sum of squares)
error_rate = 0
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    error_rate += np.sum((cluster - centroids[i]) ** 2)

# Output initial and final clusters, epoch size, and error rate
print("Initial Centroids:")
print(initial_centroids)
print("\nFinal Centroids:")
print(centroids)
print("\nEpochs to Converge:", epochs)
print("Error Rate (Within-Cluster Sum of Squares):", error_rate)

# Plotting the results
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i + 1}")

plt.scatter(centroids[:, 0], centroids[:, 1], color="red", marker="x", s=100, label="Final Centroids")
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], color="blue", marker="o", s=100, alpha=0.5, label="Initial Centroids")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.title("K-Means Clustering")
plt.show()
