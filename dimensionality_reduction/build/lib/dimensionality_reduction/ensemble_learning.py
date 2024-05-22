
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class ensemble_learning_feature_selection:
    
    def __init__(self, estimator=RandomForestClassifier(), n_features=None):
        self.estimator = estimator
        self.n_features = n_features
        self.feature_importance = None
        self.selected_indices = None

    def fit(self, X, y,use_mda = False):
        self.estimator.fit(X, y)
        self.feature_importance = self.estimator.feature_importances_
        self.selected_indices = np.argsort(self.feature_importance)[::-1]

        if self.n_features is not None:
            self.selected_indices = self.selected_indices[:self.n_features]
        else:
            if use_mda == False:
                num = self.find_optimal_features()
            else:
                num = self.find_optimal_features_MDA(X,y)
    
        return self.selected_indices[:num].tolist() 


    def transform(self, X):
        return X[:, self.selected_indices]


    def find_optimal_features(self, use_savgol_filter=True):
        total_importance = np.sum(self.feature_importance)
        cumulative_importance = np.cumsum(self.feature_importance[self.selected_indices]) / total_importance

        if use_savgol_filter:
            window_length = min(5, len(cumulative_importance) //2 * 2 - 1) 
            polyorder = 2
            cumulative_importance = savgol_filter(cumulative_importance, window_length, polyorder)

        k = np.arange(1, len(cumulative_importance) + 1)
        knee_locator = KneeLocator(k, cumulative_importance, curve='concave', direction='increasing')

        plt.plot(k, cumulative_importance, marker='o', label='Cumulative Importance')
        if use_savgol_filter:
            plt.plot(k, cumulative_importance, label='Smoothed Cumulative Importance')


        if knee_locator.knee is  None:
            test = 0
            s = 1
            while test < 20 and knee_locator.knee is None:
                s-= 0.05
                print("Knee not found, will decrease sensitivity and try again. Νσew sensitivity =",s)
                knee_locator = KneeLocator(k, cumulative_importance, curve='concave', direction='increasing',S= s)
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
    


    def find_optimal_features_MDA(self,X,y, use_savgol_filter=True):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = self.estimator
        clf.fit(X_train, y_train)
        result = permutation_importance(clf,X_test,y_test, n_repeats=10, random_state=42)
        self.feature_importance = result.importances_mean
        sorted_indices = np.argsort(self.feature_importance)[::-1]
        negative_indices = np.where(self.feature_importance[sorted_indices] < 0)[0]

        positive_sorted_indices = sorted_indices[:len(sorted_indices) - len(negative_indices)]
        if len(negative_indices) > 0:
            print("these features have zero importance, better leave them out",negative_indices)

        sorted_indices = positive_sorted_indices
        sorted_importances = self.feature_importance[sorted_indices]
        
        total_importance = np.sum(sorted_importances)
        cumulative_importance = np.cumsum(sorted_importances) / total_importance
        self.selected_indices = sorted_indices


        if use_savgol_filter:
            window_length = min(5, len(cumulative_importance) //2 * 2 - 1) 
            polyorder = 2
            cumulative_importance = savgol_filter(cumulative_importance, window_length, polyorder)

        k = np.arange(1, len(cumulative_importance) + 1)
        knee_locator = KneeLocator(k, cumulative_importance, curve='concave', direction='increasing')

        plt.plot(k, cumulative_importance, marker='o', label='Permutation importance')
        if use_savgol_filter:
            plt.plot(k, cumulative_importance, label='Permutation importance')

        if knee_locator.knee is  None:
            test = 0
            s = 1
            while test < 20 and knee_locator.knee is None:
                s-= 0.05
                print("Knee not found, will decrease sensitivity and try again. Νσew sensitivity =",s)
                knee_locator = KneeLocator(k, cumulative_importance, curve='concave', direction='increasing',S= s)
                test += 1
                
        if knee_locator.knee is not  None:
            plt.axvline(x=knee_locator.knee, color='red', linestyle='--', label='Optimal Feature Count')
            plt.title('best features based on Permutation importance')
            plt.xlabel('Number of Features')
            plt.ylabel('Feature Importance')
            plt.legend()
            plt.grid(True)
            plt.show()
            return knee_locator.knee
        
        else:
            print("\nRecomended number of features cannot be calculated!\n")
            plt.axvline(x=0, color='red', linestyle='--', label='Optimal Feature Count')
            plt.title('best features based on Permutation importance')
            plt.xlabel('Number of Features')
            plt.ylabel('Feature Importance')
            plt.legend()
            plt.grid(True)
            plt.show()
