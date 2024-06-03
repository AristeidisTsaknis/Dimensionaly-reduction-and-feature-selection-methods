import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra

class G_MDS():

    def __init__(self, metric='euclidean',stress_type = 1):
        self.metric = metric
        self.embedding = None
        self.stress_type = stress_type


    def compute_geodesic_distances(self,X, n_neighbors=35):
        knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=True)
        geodesic_distances = dijkstra(csgraph=knn_graph, directed=False, return_predecessors=False)
    
        return geodesic_distances

    def calculate_distance_matrix(self,X, metric=2):
        if metric == 'manhattan':  
            diff = np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=-1)
        elif metric == 'euclidean':  
            diff = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
            diff = np.sqrt(diff)
        elif metric == 'geodestic':
            diff = self.compute_geodesic_distances(X=X)
        else:
           print("norm is not supported.")
           return 0
        return diff

    def double_centering(self,D):
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n))/ n
        B = -0.5 * J @ D ** 2 @ J
        return B

    def starting_config(self,D, n_components=2):
        B = self.double_centering(D)
        pca = PCA(n_components=n_components)
        X_init = pca.fit_transform(B)
        return X_init


    def kruskal_stress(self,D_highdim,D_lowdim,stress_type = 1 ):
        
        if stress_type == 0:
            num = np.sum((D_highdim - D_lowdim)** 2)
            den = np.sum(D_highdim ** 2)
            stress =np.sqrt(num /den)
            return stress
        elif stress_type ==1:
            weights = np.ones_like(D_highdim)
            num = np.sum(weights * (D_highdim - D_lowdim)**2)
            den = np.sum(D_highdim**2)
            stress = np.sqrt(num / den)
            return stress
        elif stress_type == 2:
            weights = np.ones_like(D_highdim)
            num = np.sum(weights * (D_highdim - D_lowdim)**2)
            den = np.sum(weights * D_highdim**2)
            stress = np.sqrt(num / den)
            return stress
        elif stress_type == 3:
            weights = np.ones_like(D_highdim)
            num = np.sum(weights * (D_highdim**2 - D_lowdim**2)**2)
            den = np.sum(D_highdim**4)
            stress = np.sqrt(num / den)
            return stress
        else:
            print("problem! no stress function!")


    def fit_transform(self, X,n_components):
        D_highdim = self.calculate_distance_matrix(X, metric=self.metric)
        X_init = self.starting_config(D_highdim, n_components=n_components)
        self.embedding, self.stress_all_ = self.optimize_embedding(X_init, D_highdim)

        return self.embedding


    def explore_dimensions(self, X, threshold=0.001):
        num_of_features = X.shape[1]
        stress_values = []
        last_stress = None
        finished = True
        for i in range(0, num_of_features):  
            D_highdim = self.calculate_distance_matrix(X, metric=self.metric)
            X_initial = self.starting_config(D_highdim, n_components=i+1)
            embedding, stress = self.optimize_embedding(X_initial, D_highdim)
            stress_value = stress
            stress_values.append(stress_value)  

            if last_stress is not None:
                change_in_stress = last_stress - stress_value
                if abs(change_in_stress) < threshold and stress_value < 1:
                    print("Change in stress below threshold at dimension ", change_in_stress)
                    finished = False
                    break
    
            last_stress = stress_value

        
        stress_values = np.array(stress_values)
        print("Stress values array:", stress_values)

        return stress_values,finished 



    def gradient_descent(self,X, D_highdim,learning_rate=0.001, max_iter=300, tolerance=1e-4, learning_rate_dec=0.9):
        n_samples = X.shape[0]
        stress_all = []

        for i in range(max_iter):
            D_lowdim = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
            stress = self.kruskal_stress(D_highdim, D_lowdim)
            stress_all.append(stress)

            if i > 0 and abs(stress_all[-2] - stress) < tolerance:
                break

            grad = np.zeros_like(X)
            for j in range(n_samples):
                for k in range(n_samples):
                    if j != k:
                        delta = X[j] - X[k]
                        distance = np.linalg.norm(delta)
                        if distance > 0:
                            grad[j] += delta / distance * (D_lowdim[j, k] - D_highdim[j, k])

            X -= learning_rate * grad
            learning_rate *= learning_rate_dec

        return X, stress_all
    
    def optimize_embedding(self, X_init, D_highdim):
        X_optimized, stress_history = self.gradient_descent(X_init, D_highdim, 
                                                            learning_rate=0.001, 
                                                            max_iter=100, 
                                                            tolerance=1e-3)
        
        if stress_history:
            final_stress = stress_history[-1] 
        else:
            final_stress = None

        return X_optimized, final_stress



    def find_optimal_components(self,stress_values,finished, max_dim):

        if finished is False:
            print("optimal k is ",len(stress_values))
            return
        
        dimensions = range(1, max_dim + 1)
    
        if len(stress_values) % 2 == 0:

            window_length = (len(stress_values) - 1)
            poly_order = window_length - 1
        else:
            window_length = (len(stress_values))
            poly_order = window_length - 1 
            
        smoothed_stress = savgol_filter(stress_values, window_length, poly_order)

        knee_locator = KneeLocator(dimensions, smoothed_stress, curve='concave', direction='increasing')

        if knee_locator.knee is  None:
            test = 0
            s = 1
            while test < 20 and knee_locator.knee is None:
                s-= 0.1
                print("Knee not found, will decrease sensitivity and try again. Νσew sensitivity =",s)
                knee_locator = KneeLocator(dimensions, smoothed_stress, curve='concave', direction='increasing',S= s)
                test += 1

        optimal_k = knee_locator.knee
        print(f"The optimal number of dimensions (k) after smoothing is: {optimal_k}")

        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, stress_values, 'b-', marker='o', label='Original Stress Values')
        plt.plot(dimensions, smoothed_stress, label='Smoothed Stress Values')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.title('Scree Plot of Stress Values with Optimal k Highlighted')
        plt.xlabel('Number of Dimensions (k)')
        plt.ylabel('Stress Value')
        plt.xticks(dimensions)
        plt.legend()
        plt.grid(True)
        plt.show()

        return optimal_k
    
    
