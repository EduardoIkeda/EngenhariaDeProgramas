import numpy as np
import matplotlib.pyplot as plt
from time import time

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, X):
        min_val = np.min(X)
        max_val = np.max(X)
        centroids = np.zeros((self.k, X.shape[1]))
        S = (max_val - min_val) / self.k
        ct = min_val
        for k in range(self.k):
            ct += S
            for c in range(X.shape[1]):
                centroids[k, c] = ct
        return centroids

    def assign_clusters(self, X, centroids):#FOCA[0, 2n+k+1,  nk+2n+k+1, nk+5n+k+11]
        distances = np.abs((X - centroids[:, np.newaxis]).sum(axis=2))      #[0, n+k+1, n+k+1,   n+k+9]
        return np.argmin(distances, axis=0)                                 #[0,     n,  n+nk, nk+4n+2]

    def update_centroids(self, X, clusters):#FOCA[0, 5nk+2k+1, 4nk + 2k, 16kn + 12k + 5]         [0, nk, nk, nk]
        centroids = - np.ones((self.k, X.shape[1]))                                          #[0, k+1,   k,      3k+3]  

    #np.ones((self.k, X.shape[1]))          #  [0, k, k, 3k+2]
    #vetor[k] = []
    #for i in range(k):                     #  [0, k, k, k+2]
    #   vetor[i] = 1                        # k[0, 0, 0,   2]    # [0, 0, 0, 2k]

        for i in range(self.k):                                                              #[0,   k,   k,       k+2]
            if np.any(clusters == i):                            #k[0, n,  2n,  4n+2]        #[0,  nk, 2nk,   4nk +2k]

    #np.any(clusters == i)                  #FOCA: [0, n,  2n,  4n+2]
    #for i in range(n):                     #  [0, n, n, n+2]
    #   if(cluster[i] == valor)             # n[0, 0, 1,   3]   #[0, 0, n, 3n]
    #       return true
    # return false
                centroids[i] = np.mean(X[clusters == i], axis=0) #k[0, 4n, 2n, 12n + 6]      #[0, 4nk, 2nk, 12kn + 6k]
    #np.mean(X[clusters == i], axis=0)      #FOCA [0, 4n, 2n, 12n + 6]
    #for i in range(n):                     #  [0, n, n, n+2]
    #   somatorio = 0                       # n[0, 0, 0,   1]   #[0, 0, 0, n]
    #   contador = 0                        # n[0, 0, 0,   1]   #[0, 0, 0, n]
    #   if (X[i] == valor)                  # n[0, 0, 1,   3]   #[0, 0, n, 3n]
    #       somatorio = somatorio + x[i]    # n[0, 1, 0,   4]   #[0, n, 0, 4n]
    #       contador = contador + 1         # n[0, 1, 0,   2]   #[0, n, 0, 2n]
    # return somotario / contador            # n[0, 1, 0,   2]   #[0, n, 0, 2]
        return centroids

    def fit(self, X):     
        self.centroids = self.initialize_centroids(X)
        tic = time()
        for _ in range(self.max_iters):                                         
                                                                            #FOCA[0, 5nk+2n+4k+2, 5nk+2n+5k+1, 17nk+5n+18k+21] => [0, nk, nk, nk]
            clusters = self.assign_clusters(X, self.centroids)              #FOCA[0,   2n+k+1, nk+2n+k+1,     nk+5n+k+11]
            new_centroids = self.update_centroids(X, clusters)              #FOCA[0, 5nk+2k+1,  4nk + 2k, 16nk + 12k + 8]         [0, nk, nk, nk]
            if np.all(self.centroids == new_centroids):                     #FOCA[0,        k,        2k,         5k + 2]

    #np.all(self.centroids == new_centroids)#  [0, k, 2k, 5k + 2]
    #for i in range(k):                     #  [0, k, k, k+2]
    #   if(centroid[i] != new_centroid[i])  # k[0, 0, 1,   4]    [0, 0, k, 4k]
    #       return false
    #return true
                break
            self.centroids = new_centroids                                    #[0, 0, 0, 2]
                
        return clusters, self.centroids, tic
    
    #Temos que o tempo por iteração é:
    #TempoporIteracao = to * (5nk+2n+4k+2) + tc * (5nk+2n+5k+1) + ta * (17nk+5n+18k+21)

    #Considerando tempo médio igual das primitivas tempos que to = tc  = ta = t (tempo médio)
    # 27nk+9n+27k+24

    def plot(self, X, clusters):
        plt.scatter(X, clusters, c=clusters, cmap='viridis')
        plt.title('KMeans Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Cluster')
        plt.show()

    def plot_clusters(self, image, clusters):
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
        plt.title('Clusters e Intensidade dos Pixels')
        plt.xlabel('Índice do Pixel')
        plt.ylabel('Intensidade do Pixel')
        plt.colorbar(label='Cluster')
        
        # Calcular os valores máximos e mínimos para cada cluster
        valores_maximos = [np.max(flat_image[flat_clusters == i]) for i in range(len(np.unique(flat_clusters)))]
        valores_minimos = [np.min(flat_image[flat_clusters == i]) for i in range(len(np.unique(flat_clusters)))]
        
        # Juntar os valores máximos e mínimos em um único array
        valores_clusters = np.concatenate((valores_maximos, valores_minimos))
        
        for valor in valores_clusters:
            plt.axhline(y=valor, color='red', linestyle='-')
        
        
        for valor in self.centroids:
            plt.axhline(y=valor, color='black', linestyle='--')
        
        plt.show()

        image = (image * 255).astype(np.uint8)

        plt.hist(np.array(image).ravel(), bins=64, range=(0, 256), color='gray', alpha=0.7)

        for valor in valores_clusters:
            plt.axvline(x=valor * 255, color='red', linestyle='-')
            
        for valor in self.centroids:
            plt.axvline(x=valor * 255, color='black', linestyle='--')
        
        plt.title('Histograma de Intensidades de Cinza')
        plt.xlabel('Intensidade de Cinza')
        plt.ylabel('Frequência')
        plt.show()
