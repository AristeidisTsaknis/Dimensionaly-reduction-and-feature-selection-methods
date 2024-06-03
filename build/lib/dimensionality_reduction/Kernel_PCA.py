import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter
import warnings
from sklearn.preprocessing import StandardScaler

class Kernel_PCA():
    def __init__(self, kernel_type='linear', gamma=None, coef0=1, degree=3):
        self.eigenvectors = None
        self.eigenvalues = None
        self.transformed_data = None
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree


    def kernel_functions(self, X, y):
        if self.kernel_type == 'linear':
            return np.dot(X, y)
        elif self.kernel_type == 'polynomial':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[0] 
            return (self.gamma * np.dot(X, y) + self.coef0) ** self.degree
        elif self.kernel_type == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[0]
            return np.exp(-self.gamma * np.linalg.norm(X - y) ** 2)
        elif self.kernel_type == 'sigmoid':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[0]
            return np.tanh(self.gamma * np.dot(X, y) + self.coef0)
        else:
            warnings.warn('Please provide a valid kernel')

            

    def calc_gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X 
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        gram_matrix = np.zeros((n_samples_X, n_samples_Y))
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                gram_matrix[i, j] = self.kernel_functions(X[i], Y[j])
        return gram_matrix

    
    
    def fit_transform(self, X, k = None,use_savgol_filter=True):
        features_number = X.shape[1]
        self.train_data = np.array(X, copy=True) 
        gram_matrix = self.calc_gram_matrix(X)
        gram_matrix_centered = self.center_gram_matrix(gram_matrix)
        self.eigenvectors, self.eigenvalues = self.calc_eigenvector_eigenvalues(gram_matrix_centered)
        self.eigenvectors, self.eigenvalues = self.sort_eigenvectors_eigenvalues(self.eigenvalues, self.eigenvectors)

        if k is None:
            k = self.find_optimal_n_components(self.eigenvalues,features_number,use_savgol_filter=use_savgol_filter)
            return k
        if k < 1:
            k = self.find_k_based_on_variance_rate(self.eigenvalues, k)
        if len(self.eigenvalues) < k:
            raise ValueError("K must be smaller than the number of attributes that the dataset has.")
        
        self.eigenvectors = self.eigenvectors[:, :k]
        self.eigenvalues = self.eigenvalues[:k]
        self.transformed_data = self.transform_data(gram_matrix_centered)
        return self.transformed_data

    

    def transform(self,X):
        gram_matrix = self.calc_gram_matrix(X,self.train_data)
        gram_matrix_centered = self.center_gram_matrix(gram_matrix,training=False)
        non_zeros = np.flatnonzero(self.eigenvalues)
        scaled_alphas = np.zeros_like(self.eigenvectors)
        scaled_alphas[:, non_zeros] = self.eigenvectors[:, non_zeros] / np.sqrt(self.eigenvalues[non_zeros])

        X_new = np.dot(gram_matrix_centered, scaled_alphas)

        return X_new


    def center_gram_matrix(self, K, training=True):
        if training:
            row_means = np.mean(K, axis=1, keepdims=True)
            column_means = np.mean(K, axis=0, keepdims=True)
            total_mean = np.mean(K)
            K_centered = K - row_means - column_means + total_mean
         
            self.train_row_means = row_means
            self.train_column_means = column_means
            self.train_total_mean = total_mean
        else:
            row_means = np.mean(K, axis=1, keepdims=True)  
            K_centered = K - row_means - self.train_column_means + self.train_total_mean

        return K_centered
        
    def standardize_data(self,dataset):
        dataset = StandardScaler().fit_transform(dataset)
        return dataset


    def calc_eigenvector_eigenvalues(self,dataset):
        eigenvalues, eigenvectors = np.linalg.eig(dataset)
        return eigenvectors, eigenvalues

    def discard_im_part(self,eigenvalues,eigenvectors):
        eigenvalues_real = np.real(eigenvalues)
        eigenvectors_real = np.real(eigenvectors)
        return eigenvectors_real,eigenvalues_real
    
    def sort_eigenvectors_eigenvalues(self,eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return sorted_eigenvectors,sorted_eigenvalues


    def find_k_based_on_variance_rate(self,eigenvalues,variance_rate):

        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var
        k=1

        while cumulative_var[k - 1] < variance_rate:
            k += 1
        return k
    
    def transform_data(self,gram_matrix):
        eigenvectors = self.eigenvectors/ np.sqrt(self.eigenvalues)
        transformed_data = np.dot(gram_matrix, eigenvectors)
        transformed_data = transformed_data.real
        return transformed_data
        
    
    def find_optimal_n_components(self,eigenvalues,features_number, use_savgol_filter=False):
        eigenvalues = eigenvalues[:features_number]
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
    
