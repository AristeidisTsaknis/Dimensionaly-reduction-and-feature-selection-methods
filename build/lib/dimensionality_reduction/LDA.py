
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter
from sklearn.preprocessing import OrdinalEncoder

class LDA:
    
    def __init__(self):
        self.eigenvectors = None
        self.eigenvalues = None
        self.transformed_data = None


    def fit_transform(self, dataset, labels, k = None):

        if not self.is_data_numerical(dataset):
            print("data must me numerical!")
            return
        
        if k is not None:
            if k > min(dataset.shape[1], len(np.unique(labels))) - 1:
                print("k must be smaller than the number of features -1")
                return
        

        if not np.issubdtype(np.array(labels).dtype, np.number):
            encoder = OrdinalEncoder()
            labels = encoder.fit_transform(np.array(labels).reshape(-1, 1))
            labels = labels.ravel()
            
    
        S_W = self.calc_within_class_scatter_matrix(dataset, labels)
        S_B = self.calc_between_class_scatter_matrix(dataset, labels)

        S_W_inv = np.linalg.pinv(S_W)
        matrix = np.dot(S_W_inv, S_B)

        self.eigenvectors,  self.eigenvalues = self.calc_eigenvector_eigenvalues(matrix)
        self.eigenvectors,  self.eigenvalues = self.sort_eigenvectors_eigenvalues( self.eigenvalues,  self.eigenvectors)


        if k is None:
            return self.find_optimal_components()
        
        if k < 1:
            k = self.find_k_based_on_discriminant_power_rate(k,dataset.shape[1],labels)

        self.eigenvectors = self.eigenvectors[:,:k]
        self.eigenvalues =  self.eigenvalues[:k]

        self.transformed_data = self.transform( dataset)
        return self.transformed_data


    def calc_eigenvector_eigenvalues(self, dataset):
        eigenvalues, eigenvectors = np.linalg.eig(dataset)

        return eigenvectors, eigenvalues

    def calc_within_class_scatter_matrix(self, features, labels):

        class_labels = np.unique(labels)
        n_features = features.shape[1]
        S_W = np.zeros((n_features, n_features))


        for label in class_labels:
            class_samples = features[labels == label]
            class_mean = np.mean(class_samples, axis=0)

            deviations = class_samples - class_mean
            covariance_matrix = np.dot(deviations.T, deviations)

            S_W += covariance_matrix
        

        return S_W

    def calc_between_class_scatter_matrix(self, features, labels):

        overall_mean = np.mean(features, axis=0)
        class_labels = np.unique(labels)
        n_features = features.shape[1]
        S_B = np.zeros((n_features, n_features))


        for label in class_labels:
            class_samples = features[labels == label]
            class_mean = np.mean(class_samples, axis=0)
            n = len(features[labels == label])

            mean_vec = class_mean.reshape(n_features, 1) 
            overall_mean_vec = overall_mean.reshape(n_features, 1)  
            S_B += n * np.dot((mean_vec - overall_mean_vec), (mean_vec - overall_mean_vec).T)
            
        return S_B



    def sort_eigenvectors_eigenvalues(self, eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        return sorted_eigenvectors, sorted_eigenvalues



    def is_data_numerical(self, dataset):
        return np.issubdtype(dataset.dtype, np.number)
    


    def transform(self, standardize_dataset):
        if self.eigenvectors.shape[0] != standardize_dataset.shape[1]:
            print("Number of features in eigenvectors must match the number of columns in the dataset.")
            return 

        transformed_data = np.dot(standardize_dataset, self.eigenvectors)
        return transformed_data


    def find_k_based_on_discriminant_power_rate(self,discriminant_power_rate,n_features,class_labels):
        total = sum(self.eigenvalues)
        explained_variance_ratio = [(i / total) for i in sorted(self.eigenvalues, reverse=True)]
        k = 1
        while k <= len(explained_variance_ratio) and explained_variance_ratio[k - 1] < discriminant_power_rate:
            if k > min(n_features, len(np.unique(class_labels))) - 1:
                k = k - 1
                break
            else:
                k += 1

        return k
    


    def find_optimal_components(self,use_savgol_filter = True):
        total = sum(self.eigenvalues)
        explained_variance_ratio = [(i / total) for i in sorted(self.eigenvalues, reverse=True)]
        cumulative_var = np.cumsum(explained_variance_ratio)
        cumulative_var= cumulative_var.real
        window_length = len(cumulative_var)
        
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
        

