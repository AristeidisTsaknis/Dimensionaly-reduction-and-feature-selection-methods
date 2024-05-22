
import numpy as np

class Spearman_Rank_Correlation:
    def __init__(self):
        pass



    def calculate_rank(self, data):
        data_np = np.array(data)
        sorted_indices = np.argsort(data_np)
        sorted_data = data_np[sorted_indices]
        ranks = np.zeros(len(data_np))


        i = 0
        while i < len(data_np):
            j = i
            while j + 1 < len(data_np) and data_np[sorted_indices[j]] == data_np[sorted_indices[j + 1]]:
                j += 1

            avg_rank = 1 + (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1

        return ranks

    
    def correlation_coefficient(self, x, y):

        if len(x) != len(y):
            print("x and y must have the same length")
        
        rank_x = self.calculate_rank(x)
        rank_y = self.calculate_rank(y)
        
    
        squared_sum = np.sum((rank_x - rank_y) ** 2)
        n = len(x)
        rank_coeff = 1 - (6 * squared_sum) / (n * (n ** 2 - 1))
        
        return rank_coeff
    

    def feature_selection(self,X,y, threshold=None):
 
        selected_features = []
        correlations = []
        target = y  
        if threshold is None:
            threshold = 0
        for i in range(X.shape[1]):
            feature = X[:, i]
            feature_cor = self.correlation_coefficient(feature, target)
            if abs(feature_cor) > threshold:
                selected_features.append(i)
                correlations.append(f"correlation for feature {i}: "+str(feature_cor))

        print("correlation",selected_features)

        return selected_features
    

    def calculate_feature_correlations(self,features):
        n = features.shape[1]
        correlations = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n): 
                if i == j:
                    correlations[i, j] = 1.0 
                else:
                    coef = self.correlation_coefficient(features[:, i], features[:, j])
                    correlations[i, j] = coef
                    correlations[j, i] = coef 
        return correlations
    
