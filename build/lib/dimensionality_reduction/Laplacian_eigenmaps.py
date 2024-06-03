import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class laplacian_eigenmaps():
     
    def __init__(self):
            self.embeddings = None
            self.w = None
            print("initialized")


    def compute_laplacian(self,W,normalize="symmetric"):
        W= np.array(W)
        degrees = W.sum(axis=1)
        
        D = np.diag(degrees)
        L = D - W
        
        if normalize == "symmetric":
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        elif normalize == "rw":
            D_inv = np.diag(1.0 / degrees)
            L = np.dot(D_inv, L)
        
        return L
    

    def laplacian_eigenmaps(self,L,num_of_features, n_components = None, find_best_num_of_components = False):
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        if find_best_num_of_components:
            n_components = self.find_optimal_n_components_spectral(eigenvalues,num_of_features,True)

        embedding = eigenvectors[:, 1:n_components+1]
        return embedding
    

    def fit_tranform(self,X, n_neighbors, n_components = None, normalize='none',find_best_num_of_components = False):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        features = X.shape[1]
        W = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False, mode='connectivity').toarray()
       
        W =  0.5 * (W + W.T)
        self.w = W
        L = self.compute_laplacian(W,normalize)
        embedding = self.laplacian_eigenmaps(L,n_components = n_components,find_best_num_of_components = find_best_num_of_components,num_of_features = features)

        return embedding
    



    def find_optimal_n_components(self,eigenvalues,num_of_features, use_savgol_filter=False):

       
        eigenvalues = eigenvalues[1:num_of_features+1]
        
        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var

        if use_savgol_filter:
            window_length = min(5, len(cumulative_var) // 2 * 2 - 1)
            polyorder = window_length - 1
            cumulative_var = savgol_filter(cumulative_var, window_length, polyorder)

        k = np.arange(1, len(cumulative_var) + 1)
        knee_locator = KneeLocator(k, cumulative_var, curve='convex', direction='increasing')

        plt.plot(k, cumulative_var, marker='o', label='Cumulative Importance')
        if use_savgol_filter:
            plt.plot(k, cumulative_var, label='Smoothed Cumulative Importance')

        if knee_locator.elbow is None:
            s = 1
            test = 0
            while test < 20 and knee_locator.knee is None:
                s -= 0.05
                print("Knee not found, will decrease sensitivity and try again. New sensitivity =", s)
                knee_locator = KneeLocator(k, cumulative_var, curve='convex', direction='increasing',S= s)
                test += 1

        if knee_locator.elbow is not None:
            plt.axvline(x=knee_locator.elbow, color='red', linestyle='--', label='Optimal Feature Count')
            plt.title('Optimal Features for Spectral Embedding')
            plt.xlabel('Number of Features')
            plt.ylabel('Feature Importance')
            plt.legend()
            plt.grid(True)
            plt.show()

        if knee_locator.elbow is None:
            print("could not find an optimal number of dimensions. Manual inspection is recomended")   
            return 1 
        else:
            print("Recommended number of dimensions:", knee_locator.elbow)

            return knee_locator.elbow
        
