import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter

class PCA:
    
    def __init__(self):
        self.eigenvectors = None
        self.eigenvalues = None
        self.transformed_data = None
        
    def fit_transform(self,dataset,k = None):
              
        if not self.is_data_numerical(dataset):
            print("data must me numerical!")
            return

        dataset = self.standardize_data(dataset)
        cov_matrix =self.calc_covariance_matrix(dataset)
        eigenvectors,eigenvalues = self.calc_eigenvector_eigenvalues(cov_matrix)
        eigenvectors,eigenvalues  = self.sort_eigenvectors_eigenvalues(eigenvalues, eigenvectors)

        if k is None:
            return self.find_optimal_n_components(eigenvalues)
        if k < 1:
            k = self.find_k_based_on_variance_rate(eigenvalues,k)

        if len(eigenvalues) < k:
            print("K must be smaller than the number of attributes that the dataset has")
            return

        self.eigenvectors = eigenvectors[:, :k]
        self.eigenvalues = eigenvalues[:k]
        self.transformed_data = self.transform(dataset)
        return self.transformed_data

    def standardize_data(self,dataset):
        mean = np.mean(dataset, axis=0)
        std_deviation = np.std(dataset, axis=0)
        std_deviation = np.where(std_deviation != 0, std_deviation, 1.0)
        std_dataset =(dataset - mean)/std_deviation
        return std_dataset

    def calc_covariance_matrix(self,data):
        # rowvar = false -> columns are the attributes
        covariance_matrix = np.cov(data, rowvar=False)
        return covariance_matrix


    def calc_eigenvector_eigenvalues(self,dataset):
        eigenvalues, eigenvectors = np.linalg.eig(dataset)
        return eigenvectors, eigenvalues

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

    
    def is_data_numerical(self,dataset):
        return np.issubdtype(dataset.dtype, np.number)

    def transform(self,standardize_dataset):

        if self.eigenvectors.shape[0] != standardize_dataset.shape[1]:
            print("Number of features in eigenvectors must match the number of columns in the dataset.")
            return 
        
        transformed_data = np.dot(standardize_dataset, self.eigenvectors)
        return transformed_data

    def find_optimal_n_components(self,eigenvalues, use_savgol_filter=True):
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
    




from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris = load_iris()
X = iris.data

x_scaled = StandardScaler().fit_transform(X)



my_pca = PCA()
my_pca.fit_transform(x_scaled)
