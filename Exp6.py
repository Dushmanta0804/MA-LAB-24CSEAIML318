#question 1
import numpy as np
from sklearn.cluster import KMeans


points = np.array([[2,3], [3,4], [6,6], [7,7]])


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(points)


labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster labels for each point:", labels)
print("Centroids of clusters:\n", centroids)

# visualisation Graph
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

points = np.array([[2,3], [3,4], [6,6], [7,7]])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(points)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', marker='o')


plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title("KMeans Clustering Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

+#QUESTION 2
import numpy as np
from sklearn.cluster import KMeans


points = np.array([[2],[4],[10],[12],[3],[20],[30],[11],[25]])

kmeans = KMeans(n_clusters=2, init=np.array([[2],[20]]), n_init=1, random_state=42)
kmeans.fit(points)


labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster labels for each point:", labels)
print("Centroids of clusters:\n", centroids)


# visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


points = np.array([[2],[4],[10],[12],[3],[20],[30],[11],[25]])


kmeans = KMeans(n_clusters=2, init=np.array([[2],[20]]), n_init=1, random_state=42)
kmeans.fit(points)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


plt.scatter(points, np.zeros_like(points), c=labels, cmap='viridis', marker='o', s=100)


plt.scatter(centroids, np.zeros_like(centroids), c='red', marker='X', s=200, label='Centroids')

plt.title("KMeans Clustering (1D Data)")
plt.xlabel("Value")
plt.yticks([])  
plt.legend()
plt.show()

