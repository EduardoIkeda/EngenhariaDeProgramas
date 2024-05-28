import numpy as np
import matplotlib.pyplot as plt

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

    def assign_clusters(self, X, centroids):
        distances = np.abs((X - centroids[:, np.newaxis]).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, clusters):
        # centroids = np.zeros((self.k, X.shape[1])) 
        # DIEGO: inicializar com 0 não é bom, pq zero é 
        # um valor válido pode ter valores negativos que não tem sentido e aí dá para detectar 
        # facilmente, por exemplo -1 a seguir 
        centroids = - np.ones((self.k, X.shape[1]))
        for i in range(self.k):
            if np.any(clusters == i):
                centroids[i] = np.mean(X[clusters == i], axis=0)
            # DIEGO: controle adiantado de cluster vazio
            else: 
                print(">>>>>>>>>>>>>> cluster",i,"vazio, call IKEDA")
                centroids[i] = X[np.random.choice(X.shape[0])] 
        return centroids

    def fit(self, X):
        # DIEGO: use uma variavel de controle de execução
        fit_OK = True
        numero_iteracoes_realizadas = 0
        
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, clusters)
            # DIEGO controle de cluster vazio
            if np.any(new_centroids == -1): # DIEGO
                print("============> Tem clusters vazio") # DIEGO
                fit_OK = False # DIEGO
            elif np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
            numero_iteracoes_realizadas = numero_iteracoes_realizadas + 1
                
        return clusters, self.centroids, fit_OK, numero_iteracoes_realizadas  # DIEGO retorne a variavel de controle

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

        plt.hist(np.array(image).ravel(), bins=64, range=(0, 256), color='gray', alpha=0.7)

        for valor in valores_clusters:
            plt.axvline(x=valor, color='red', linestyle='-')
            
        for valor in self.centroids:
            plt.axvline(x=valor * 255, color='black', linestyle='--')
        
        plt.title('Histograma de Intensidades de Cinza')
        plt.xlabel('Intensidade de Cinza')
        plt.ylabel('Frequência')
        plt.show()