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

    def assign_clusters(self, X, centroids):#FOCA[0, n, nk, nk]
        distances = np.abs((X - centroids[:, np.newaxis]).sum(axis=2))      #[0, n+k+1, n+k+1,   n+k+9]
        return np.argmin(distances, axis=0)                                 #[0,     n,  n+nk, nk+4n+2]

    def update_centroids(self, X, clusters):#FOCA[0, nk, nk, nk]
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
        fit_OK = True       
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
                                                                          #FOCA[0, nk + 2n, 2nk + 2n, 2nk + 5n + 6] => [0, nk, nk, nk]
            clusters = self.assign_clusters(X, self.centroids)                #[0,  n, nk, nk + 1]
            new_centroids = self.update_centroids(X, clusters)                #[0, nk, nk, nk + 1]
            if np.all(self.centroids == new_centroids):                       #[0,  n, 2n, 5n + 2]

    #np.all(self.centroids == new_centroids)#  [0, n, 2n, 5n + 2]
    #for i in range(n):                     #  [0, n, n, n+2]
    #   if(centroid[i] != new_centroid[i])  # n[0, 0, 1,   4]    [0, 0, n, 4n]
    #       return false
    #return true
                break
            self.centroids = new_centroids                                    #[0, 0, 0, 2]
                
        return clusters, self.centroids, fit_OK