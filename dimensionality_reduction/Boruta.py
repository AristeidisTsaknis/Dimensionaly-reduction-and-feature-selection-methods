from sklearn.utils import shuffle
from scipy.stats import binomtest
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

class Boruta:

    def __init__(self,estimator = None,perc = 100):

        if estimator == None:
            self.estimator = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.estimator = estimator
        self.perc = perc
        self.support_ = np.array([])
        self.ranking = np.array([])
        self.feature_importances = np.array([])


    def fit(self,X,y,max_iter=100,p=0.5,alpha=0.05,two_step = False):
        n_features = X.shape[1]
        self.feature_importances = np.zeros(n_features)
        self.ranking = np.zeros(n_features,dtype=int)
        
        for i in range(0,max_iter):
            X_updated = self.create_shadow_features(X=X)
            importances = self.train_tree_estimator(X_updated,y)
            shadow_feature_indices = np.arange(n_features)+n_features

            self.feature_importances += importances[:n_features]
            shadow_importances = importances[shadow_feature_indices]
            self.update_ranking(shadow_importances,i)
        
        if two_step:
            significant_feautures = self.binomial_test_with_bh_correction(max_iter,p,alpha)
        else:
            significant_feautures = self.binomial_test_with_bonferroni_correction(max_iter,p,alpha)

        return significant_feautures


    def create_shadow_features(self,X):
            X_shadow = shuffle(X)
            X_updated = np.hstack((X, X_shadow))
            return X_updated
    

    def train_tree_estimator(self,X,y):

        estimator  = clone(self.estimator)
        estimator.fit(X, y)
        return estimator.feature_importances_

    def update_ranking(self,shadow_importances,iter):
        
        threshold = np.percentile(shadow_importances, self.perc)
        for i in range(len(self.feature_importances)):
                if self.feature_importances[i]/(iter + 1) > threshold:
                    self.ranking[i] += 1 


    def binomial_test_with_bonferroni_correction(self,max_iter, p=0.5, alpha=0.05):
        significant_features = []
        p_values = []

        for i, successes in enumerate(self.ranking):
            result = binomtest(successes, max_iter, p=p, alternative='greater')
            p_values.append(result.pvalue)


        p_values = np.array(p_values)
        adjusted_alpha = alpha / len(p_values)
        
  
        for i, p_value in enumerate(p_values):
            if p_value < adjusted_alpha:
                significant_features.append(i)
        
        return significant_features
    
    def binomial_test_with_bh_correction(self, max_iter, p=0.5, alpha=0.05):
        significant_features = []
        p_values = []

        for i, successes in enumerate(self.ranking):
            result = binomtest(successes, max_iter, p=p, alternative='greater')
            p_values.append((i, result.pvalue))

        p_values.sort(key=lambda x: x[1])

        m = len(p_values)  
        prev_bh_value = 1  
        for i, (original_index, p_value) in enumerate(p_values):
            bh_threshold = (i + 1) / m * alpha
            if p_value <= bh_threshold:
                prev_bh_value = p_value 
                significant_features.append(original_index)
            elif p_value > bh_threshold:
                break  

        significant_features = [idx for idx, p in p_values if p <= prev_bh_value]

        return sorted(significant_features)
    

    