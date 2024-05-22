from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter
from numpy.random import RandomState

class Locally_Linear_Embedding:
    def __init__(self,n_neighbors=12,n_components = None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.embedding = None
        self.nbrs = None


    def fit_transform(self, X, reg=1e-3):
        W = self.barycenter_kneighbors_graph(X, self.n_neighbors, reg=reg, n_jobs=-1)
        eigenvalues, eigenvectors = self.compute_embedding(X=X, W=W, n_components=self.n_components)

        if self.n_components is None:
            num_of_features = X.shape[1]
            eigenvalues = eigenvalues[1:num_of_features + 1]
            self.n_components = self.find_optimal_components(eigenvalues, True)
            return self.n_components
        

        self.embedding = eigenvectors[:, 1:self.n_components + 1]

        return self.embedding


    def barycenter_kneighbors_graph(self,X, n_neighbors, reg=1e-3, n_jobs=None):
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
        X = knn._fit_X
        self.nbrs = knn
        n_samples = X.shape[0]
        ind = knn.kneighbors(X, return_distance=False)[:, 1:] 
        Y_neighbors = X[ind]
        data = self.barycenter_weights(X, Y_neighbors, reg=reg)
        indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
        return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))

    

    def compute_embedding(self, X, W, n_components,random_state = 42):
        n_samples = X.shape[0]
        I = eye(n_samples, format='csr') 
        I_minus_W = I - W

        
        if not isinstance(I_minus_W, csr_matrix):
            I_minus_W = csr_matrix(I_minus_W)

        M = I_minus_W.T.dot(I_minus_W) 

        if n_components is None:
            n_components = X.shape[1]

        k = n_components
        k_skip = 1
        random_state = RandomState(random_state)
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_vals, eigen_vect = eigsh(M, k + k_skip, sigma=0.0, tol=1e-6, maxiter=100, v0=v0)
        except RuntimeError as e:
            raise ValueError(
                "Error in determining null-space with ARPACK."
            ) from e

        return eigen_vals, eigen_vect


    def find_optimal_components(self,eigenvalues,use_savgol_filter=False):
        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var 

        if use_savgol_filter:
            window_length = min(5, len(cumulative_var) //2 * 2 - 1) 
            polyorder = 2
            cumulative_var = savgol_filter(cumulative_var, window_length, polyorder)

        k = np.arange(1, len(cumulative_var) + 1)
        knee_locator = KneeLocator(k, cumulative_var, curve='convex', direction='increasing')

        plt.plot(k, cumulative_var, marker='o', label='Cumulative Importance')
        if use_savgol_filter:
            plt.plot(k, cumulative_var, label='Smoothed Cumulative Importance')


        if knee_locator.knee is  None:
            test = 0
            s = 1
            while test < 20 and knee_locator.knee is None:
                s-= 0.05
                print("Knee not found, will decrease sensitivity and try again. Νσew sensitivity =",s)
                knee_locator = KneeLocator(k, cumulative_var, curve='convex', direction='increasing',S= s)
                test += 1
                
        if knee_locator.knee is not  None:
            plt.axvline(x=knee_locator.knee, color='red', linestyle='--', label='Optimal Feature Count')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('Cumulative Variance Explained by LLE Components')
            plt.legend()
            plt.grid(True)
            plt.show()

        return knee_locator.knee
    


    def transform(self, X_new, reg=1e-3):
        distances, indices = self.nbrs.kneighbors(X_new, n_neighbors=12)
        neighbors = self.nbrs._fit_X[indices]
        weights = self.barycenter_weights(X_new, neighbors, reg=reg)
        X_transformed = np.zeros((X_new.shape[0], self.n_components))
        for i in range(X_new.shape[0]):
            X_transformed[i] = np.dot(weights[i], self.embedding[indices[i]])

        return X_transformed

    def barycenter_weights(self, X, neighbors, reg=1e-3):
        n_samples = X.shape[0]
        n_neighbors = neighbors.shape[1]  
        weights = np.zeros((n_samples, n_neighbors))
        for i in range(n_samples):
            Z = neighbors[i] - X[i] 
            C = np.dot(Z, Z.T)  

            if np.issubdtype(C.dtype, np.integer) and isinstance(reg, float):
                C = C.astype('float64') 
            C += np.eye(C.shape[0]) * reg 
            w = np.linalg.solve(C, np.ones(n_neighbors)) 
            weights[i] = w / w.sum()
        return weights
    
