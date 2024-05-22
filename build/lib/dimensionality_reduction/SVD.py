import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.signal import savgol_filter


class svd():
    def __init__(self):
        self.U = None  
        self.sigma = None  
        self.VT = None  
        self.transformed_data = None


    def fit_transform(self,dataset,k = None):

        if not self.is_data_numerical(dataset):
            print("data must me numerical!")
            return
        
        self.calc_VT_U_Sigma(dataset)

        if k is None:
            return self.find_optimal_n_components(self.sigma)
        if len(self.sigma) < k:
            print("K must be smaller than the number of attributes that the dataset has")
            return
        
        if k < 1:
            k = self.find_k_based_on_variance_rate(self.sigma,k)
        

        self.sigma = self.sigma[:k]
        self.U = self.U[:, :k]
        self.VT = self.VT[:k, :]
        self.transformed_data = self.transform(dataset)
        return self.transformed_data


    def find_k_based_on_variance_rate(self,eigenvalues,variance_rate):

        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var
        k=1

        while cumulative_var[k - 1] < variance_rate:
            k += 1
        return k
    



    def calc_VT_U_Sigma(self,dataset):

        self.U, sigma, self.VT = np.linalg.svd(dataset, full_matrices=False)  # Compute the SVD
        #self.sigma = np.diag(sigma)
        self.sigma = sigma
        #Den leitourgei se megala dataset o mathimatikos typos 
        #dataset = np.array(dataset)
        #transpose = dataset.T
        #A * A^T
        #prod = np.dot(dataset,transpose)
        #prod += np.eye(prod.shape[0]) * 1e-10

        #cond_number = np.linalg.cond(prod)
        #print("Condition number:", cond_number)

        #eigenvalues,U = np.linalg.eig(prod)


        # V = transose(array)*array , V^T = transpose(V)
        #prod = np.dot(transpose,dataset)
        #eigenvalues,V = np.linalg.eig(prod)

        #eigenvalues = np.sqrt(eigenvalues)
        
        #self.sigma = eigenvalues
        #self.U = U
        #self.VT = V
    
    

    def sort_matrices(self):
        indices = np.argsort(self.sigma)[::-1] 
        
        self.sigma = self.sigma[indices]
        self.VT= self.VT[:,indices]
        self.U = self.U[:,indices]


    def is_data_numerical(self,dataset):
        return np.issubdtype(dataset.dtype, np.number)
    

    def transform(self,dataset):
        transformed_data = np.dot(dataset,self.VT.T)
        return transformed_data


    def find_optimal_n_components(self,eigenvalues, use_savgol_filter=True):
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
    
