import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter
from statsmodels.multivariate.factor_rotation import rotate_factors
from math import sqrt, log
from sklearn.utils.extmath import randomized_svd

class Factor_analysis:

    def __init__(self):
        self.loadings = None
        self.eigenvalues = None
        self.transformed_data = None
        self.noise_variance = None
        self.mean = None


    def calc_correlation_matrix(self,X):
        
        correlation_matrix = np.corrcoef(X, rowvar=False)
        
        return correlation_matrix


    def calc_eigenvector_eigenvalues(self,X):
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        return eigenvectors, eigenvalues
    


    def sort_eigenvectors_eigenvalues(self,eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        sorted_eigenvectors = sorted_eigenvectors / np.linalg.norm(sorted_eigenvectors, axis=0)
        return sorted_eigenvectors,sorted_eigenvalues

    def rotate_matrix(self,rotation = None):
        if rotation == None:
            return
        else:  
            _, self.loadings = rotate_factors(self.loadings, rotation)

    def standardize_data(self,dataset):
        mean = np.mean(dataset, axis=0)
        std_deviation = np.std(dataset, axis=0)
        #check zero division
        std_deviation = np.where(std_deviation != 0, std_deviation, 1.0)
        std_dataset =(dataset - mean)/std_deviation
        return std_dataset
    

    def is_data_numerical(self,dataset):
        return np.issubdtype(dataset.dtype, np.number)
    


    def transform(self,X):
        Wpsi = self.loadings / (self.noise_variance+ 1e-12)
        #X -=self.mean
        Ih = np.eye(self.loadings.shape[0])
        cov_z = np.linalg.pinv(Ih + np.dot(Wpsi, self.loadings.T))  

        tmp = np.dot(X, Wpsi.T)
        X_transformed = np.dot(tmp, cov_z)

        return X_transformed


    def fit_transform(self, X,k = None,svd_type = None,tol=1e-2,max_iter=1000,random_state =42,rotation = None):
        n_samples, n_features = X.shape
        self.mean = np.mean(X, axis=0)
        X -= np.mean(X, axis=0)
        check = False
        if k is None:
            check =True
            k = n_features

        nsqrt = sqrt(n_samples)
        llconst = n_features * log(2.0 * np.pi) + k
        var = np.var(X, axis=0)
        psi = np.ones(n_features, dtype=X.dtype)
        old_ll = -np.inf
        loglike_ = []

        for i in range(max_iter):
            sqrt_psi = np.sqrt(psi)
            if svd_type == 'lapack':
                _, s, Vt = np.linalg.svd(X / (sqrt_psi * nsqrt), full_matrices=False)
                unexp_var = self.squared_norm(s[k:])
            else:
                _, s, Vt = randomized_svd(X / (sqrt_psi * nsqrt), n_components=k, n_iter=3, random_state=random_state)
                unexp_var = self.squared_norm(X / (sqrt_psi * nsqrt)) - self.squared_norm(s)

            s **= 2
            W = np.sqrt(np.maximum(s - 1.0, 0.0))[:, np.newaxis] * Vt
            W *= sqrt_psi

            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(psi))
            ll *= -n_samples / 2.0

            loglike_.append(ll)
            if (ll - old_ll) < tol:
                break

            old_ll = ll
            psi = np.maximum(var - np.sum(W**2, axis=0), 1e-12)
        
        self.eigenvalues = s
        if check:
            k = self.find_optimal_n_components(self.eigenvalues,n_features)
            return k 

        else:
            self.noise_variance = psi
            self.loadings = W
            self.transformed_data = self.transform(X)
            self.rotate_matrix(rotation)
            return self.transformed_data
            

    def rotate_matrix(self,rotation = None):
        if rotation == None:
            return
        elif rotation == "varimax":
            rotated_loadings, _ = rotate_factors(self.loadings, method='varimax')
            return rotated_loadings
        elif rotation == "promax":
            rotated_loadings, rotation_matrix = rotate_factors(self.loadings, method='promax')
            return rotated_loadings
        elif rotation == "oblimin":
            rotated_loadings, rotation_matrix = rotate_factors(self.loadings, method='oblimin')
            return rotated_loadings
        elif rotation == "quartimax":
            rotated_loadings, rotation_matrix = rotate_factors(self.loadings, method='quartimax')
            return rotated_loadings
    
    def squared_norm(self,x):
        return np.sum(x**2)
    

    def find_optimal_n_components(self,eigenvalues,features_number, use_savgol_filter=True):
        
        has_nans = np.isnan(eigenvalues)
        has_infs = np.isinf(eigenvalues)
        eigenvalues[has_nans] = 0
        eigenvalues[has_infs] = 0

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
    

