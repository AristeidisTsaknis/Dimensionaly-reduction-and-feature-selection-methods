import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph,NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from sklearn.impute import SimpleImputer
from sklearn.decomposition import KernelPCA

class isomap():
    
    def __init__(self):
        self.nbrs = None
        self.scaler = StandardScaler()
        self.kernel_pca = None
        self.g = None


    def fit_transform(self, X, n_components = None,n_neighbors=23):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        features_num =  X.shape[1]
        
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)

        W = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False, mode='distance')
        self.distances = shortest_path(W, method='auto', directed=False)
        distances = self.distances

        if np.isinf(self.distances).any() or np.isnan(self.distances).any():
            print("graph not connected iterative process")
            n_neighbors = int(np.sqrt(X.shape[0]))
            while True:
                W = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False, mode='distance')
                self.distances = shortest_path(W, method='auto', directed=False)
                distances = self.distances
                print("number of neighbors",n_neighbors)
                if not (np.isinf(self.distances).any() or np.isnan(self.distances).any()):
                    break
                n_neighbors +=2

        n_samples = X.shape[0]
        J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        K = -0.5 * J.dot(distances ** 2).dot(J)
        

        if n_components is  None:
            self.kernel_pca = KernelPCA(n_components= features_num, kernel="precomputed",random_state=42)
            self.kernel_pca.fit(K)
            eigenvalues = self.kernel_pca.eigenvalues_
            n_components = self.find_optimal_n_components(eigenvalues,use_savgol_filter=True)
            return n_components
        else:
            self.kernel_pca = KernelPCA(n_components=n_components, kernel="precomputed",random_state=42)
            

        embedding = self.kernel_pca.fit_transform(K)
            
        return embedding
    
    def transform(self, X):
        X = self.scaler.transform(X)
        distances, indices = self.nbrs.kneighbors(X, return_distance=True)

        n_queries = distances.shape[0]
        n_train = self.nbrs.n_samples_fit_

        dtype = np.float64
        G_X = np.zeros((n_queries, n_train), dtype)
        for i in range(n_queries):
            G_X[i] = np.min(self.distances[indices[i]] + distances[i][:, None], 0)
        
        G_X **= 2
        G_X *= -0.5
        self.g  = G_X
        return self.kernel_pca.transform(G_X)
                
    

    def find_optimal_n_components(self,eigenvalues, use_savgol_filter=False):
        eigenvalues = eigenvalues[eigenvalues > 0]
        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var 

        if use_savgol_filter:
            window_length = min(5, len(cumulative_var) //2 * 2 - 1) 
            polyorder = 2
            cumulative_var = savgol_filter(cumulative_var, window_length, polyorder)

        k = np.arange(1, len(cumulative_var) + 1)
        knee_locator = KneeLocator(k, cumulative_var, curve='concave', direction='increasing')

        plt.plot(k, cumulative_var, marker='o', label='Cumulative Importance')
        if use_savgol_filter:
            plt.plot(k, cumulative_var, label='Smoothed Cumulative Importance')


        if knee_locator.knee is  None:
            test = 0
            s = 1
            while test < 20 and knee_locator.knee is None:
                s-= 0.05
                print("Knee not found, will decrease sensitivity and try again. Νew sensitivity =",s)
                knee_locator = KneeLocator(k, cumulative_var, curve='concave', direction='increasing',S= s)
                test += 1
                
        if knee_locator.knee is not  None:
            plt.axvline(x=knee_locator.knee, color='red', linestyle='--', label='Optimal Feature Count')
            plt.title('best features')
            plt.xlabel('Number of Features')
            plt.ylabel('Feature Importance')
            plt.legend()
            plt.grid(True)
            plt.show()

        return knee_locator.knee
    

