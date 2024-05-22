from sklearn.preprocessing import OrdinalEncoder
import numpy as np 

class Kendalls_Tau_Correlation:
    def __init__(self):
        pass

    def calculate_concordant_discordant(self, x, y):
        n = len(x)
        if n != len(y):
           print("features and class feature should have the same length")

        concordant = 0
        discordant = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = (x[i] - x[j]) * (y[i] - y[j])
                if diff > 0:
                    concordant += 1
                elif diff < 0:
                    discordant += 1

        return concordant, discordant


    def ties_count(self, values):
        tie_count = 0
        val_count = {}
        for value in values:
            if value in val_count:
                val_count[value] += 1
            else:
                val_count[value] = 1

        for count in val_count.values():
            if count > 1:
                tie_count += count * (count - 1) / 2
        return tie_count


    def correlation_coefficient(self, x, y):
        n = len(x)
        if n != len(y):
            print("features and class feature should have the same length")
        
        if not np.issubdtype(np.array(y).dtype, np.number):
            encoder = OrdinalEncoder()
            y = encoder.fit_transform(np.array(y).reshape(-1, 1))
            y = y.ravel()

        concordant, discordant = self.calculate_concordant_discordant(x, y)
        n1 = self.ties_count(x)
        n2 = self.ties_count(y)
        n0 = n * (n - 1) / 2
        if (n0 - n1) == 0 or (n0 - n2) == 0:
            tau = (concordant - discordant) / np.sqrt((n0 - n1 +1) * (n0 - n2))
            return tau
        if (n0 - n2) == 0:
            tau = (concordant - discordant) / np.sqrt((n0 - n1 +1) * (n0 - n2+1))
            return tau
       
        tau = (concordant - discordant) / np.sqrt((n0 - n1) * (n0 - n2))
        
        return tau

    def feature_selection(self, X,y,threshold = None):
        correlations = []
        selected_features = []
        target = y 
        if threshold is None:
            threshold = 0
        for i in range(X.shape[1]):
            feature = X[:, i]
            tau = self.correlation_coefficient(feature, target)
            if abs(tau) > threshold:
                selected_features.append(i)
                correlations.append(f"correlation for featue {i}: {tau}")

        return selected_features
    


    def calculate_feature_correlations(self, features):
        n = features.shape[1]
        correlations = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    correlations[i, j] = 1.0 
                else:
                    x = features[:, i]
                    y = features[:, j]
                    coef = self.correlation_coefficient(x, y)
                    correlations[i, j] = coef
                    correlations[j, i] = coef  
        
        return correlations
    


