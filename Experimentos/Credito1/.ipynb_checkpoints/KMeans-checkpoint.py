import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        S = 256 / self.k
        ct = 0
        for k in range(self.k):
            ct += S
            for c in range(X.shape[1]):
                centroids[k, c] = ct
        return centroids

    def assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            if np.any(clusters == i):
                centroids[i] = np.mean(X[clusters == i], axis=0)
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, clusters)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return clusters, self.centroids

    def plot(self, X, clusters):
        plt.scatter(X, clusters, c=clusters, cmap='viridis')
        plt.title('KMeans Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Cluster')
        plt.show()

    def plot_clusters(self, image, clusters):
        # Converter os clusters de volta para uma imagem
        clustered_image = np.zeros_like(image)
        for idx, (i, j) in enumerate(np.ndindex(image.shape)):
            clustered_image[i, j] = clusters[idx]

        # Transformar a matriz em um vetor de 1 dimensão
        flat_image = image.flatten()
        flat_clusters = clusters

        # Plotar a imagem original e a imagem clusterizada
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')

        axes[1].imshow(clustered_image, cmap='viridis')
        axes[1].set_title('Imagem Clusterizada')
        axes[1].axis('off')

        # Plotar os clusters em 2D
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(flat_image)), flat_image, c=flat_clusters, cmap='viridis', s=50)

        # Marcar os centróides
        step = len(flat_image) // self.k
        for i, centroid in enumerate(self.centroids):
            plt.axvline(x=i * step, color='r', linestyle='--')
            plt.scatter([i * step + (step / 2)], [centroid[2]], color='blue', marker='x', s=200, label=f'Centroid {i+1}')

        plt.title('Clusters e Intensidade dos Pixels')
        plt.xlabel('Índice do Pixel')
        plt.ylabel('Intensidade do Pixel')
        plt.colorbar(label='Cluster')
        plt.legend()
        plt.show()