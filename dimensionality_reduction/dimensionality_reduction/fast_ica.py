import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import svd

class fast_ICA:
    def __init__(self, n_components=2, tol=1e-4):
        self.n_components = n_components
        self.tol = tol
        self.mean_ = None
        self.W = None  
        self.whiten = None 
        self.whiten_solver = 'svd'
        self.Vt_ = None
        self.S_ = None

    def sym_decorrelation(self, W):
        K = W @ W.T
        s, U = np.linalg.eigh(K)
        W_new = (U @ np.diag(1.0 / np.sqrt(s)) @ U.T) @ W
        return W_new

    def fit_transform(self, data,n_components, max_iter=200,g_type='logcosh'):
        self.n_components = n_components
        n_samples, n_features = data.shape
        self.mean_ = np.mean(data, axis=0)
        X_centered = data - self.mean_


        if self.whiten_solver == 'svd':
            U, S, Vt = svd(X_centered, full_matrices=False)
            self.S_ = S[:self.n_components]
            self.Vt_ = Vt[:self.n_components, :]
            K = (U / S)[:,:self.n_components]
            X_whitened = U[:,:self.n_components] * np.sqrt(n_samples)

        elif self.whiten_solver == 'pca':
            self.pca = PCA(whiten=True, n_components=self.n_components)
            X_whitened = self.pca.fit_transform(X_centered)

        self.W = self.initialize_randomly(self.n_components, X_whitened.shape[1])
        for iteration in range(max_iter):
            W_old = self.W.copy()
            for i in range(self.n_components):
                self.W[i, :] = self.update(self.W[i, :], X_whitened,g_type)
            self.W = self.sym_decorrelation(self.W)
            if self.has_converged(W_old, self.W):
                break

        S = (self.W @ X_whitened.T).T
        return S

    def transform(self, X):

        if self.whiten_solver == 'svd':
            X_centered = X - self.mean_
            X_whitened = (X_centered @ self.Vt_.T) / self.S_
        else:
            X_centered = X - self.mean_
            X_whitened = self.whiten.transform(X_centered)
        
        S = (self.W @ X_whitened.T).T
        return S

    def has_converged(self, W_old, W_new, tol=1e-4):
        delta_W = np.linalg.norm(W_old - W_new, ord='fro')
        return delta_W < tol

    def initialize_randomly(self, n_components, random_state=None):
        rng = np.random.default_rng(seed=random_state)
        w_init = rng.normal(size=(n_components, n_components))
        w_init /= np.linalg.norm(w_init, axis=0, keepdims=True)  
        return w_init


    def update(self, wi, X_white, type='logcosh'):
        wiX = X_white @ wi

        if type == 'logcosh':
            g = np.tanh(wiX)
            g_prime = 1 - g ** 2
        elif type == 'exp':
            g = np.exp(-wiX**2 / 2)
            g_prime = wiX * g
        elif type == 'cube':
            g = wiX**3
            g_prime = 3 * wiX**2

        grad_wi = (X_white.T @ g - g_prime.mean() * wi) / X_white.shape[0]
        return grad_wi / np.linalg.norm(grad_wi)


