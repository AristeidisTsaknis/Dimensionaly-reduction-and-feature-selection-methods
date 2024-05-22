import numpy as np
class t_sne():

    def pairwise_distances(self,X):
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return D
    

    def perplexity(self, P_row):
        P_row = P_row / np.sum(P_row) 
        entropy = -np.sum(P_row * np.log2(P_row))
        perplexity = 2 ** entropy
        return perplexity


    def conditional_probabilities(self,sigma,distances_row):
        P_row = np.exp(-distances_row / (2. * sigma ** 2))
        return P_row / np.sum(P_row)

    def binary_search_perplexity(self, distances_row, perplexity,max_iter = 1000,tolerance = 1e-8,min_sigma=1e-10,max_sigma=10000):

        for i in range(max_iter):
            sigma = (min_sigma + max_sigma) / 2
            P_row = self.conditional_probabilities(sigma,distances_row)
            current_perplexity = self.perplexity(P_row)
            if current_perplexity > perplexity:
                max_sigma = sigma
            else:
                min_sigma = sigma

            if np.abs(current_perplexity - perplexity) <= tolerance:
                return sigma
        
        return sigma
    
    def compute_conditional_probabilities(self,X, perplexity=30.0):
       
        (n_samples, n_features) = X.shape
        D = self.pairwise_distances(X)
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            distances_row = D[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n_samples)))]
            sigma = self.binary_search_perplexity(distances_row, perplexity)
            P_row = np.exp(-distances_row / (2 * sigma ** 2))
            P_row /= np.sum(P_row)  
            P[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n_samples)))] = P_row

        P = (P + P.T) / (2 * n_samples)

        return P
    

    def p_conditional(self, distances, sigmas):
        e = np.exp(-distances / (2 * np.square(sigmas.reshape((-1,1)))))
        np.fill_diagonal(e, 0.)
        e += 1e-8
        return e / e.sum(axis=1).reshape([-1,1])
    
    #calculate sigmas based on perplexity
    def find_sigmas(self, distances, perplexity):
        sigmas = np.zeros(distances.shape[0])
        for i in range(distances.shape[0]):
            distances_row = distances[i, np.concatenate((np.arange(0, i), np.arange(i + 1, distances.shape[0])))]
            sigma = self.binary_search_perplexity(distances_row, perplexity)
            sigmas[i] = sigma

        return sigmas
    

    def p_joint(self, X, perp):
        dists = self.pairwise_distances(X)
        sigmas = self.find_sigmas(dists, perp)
        p_cond = self.p_conditional(dists, sigmas)
        P = (p_cond + p_cond.T) / (2. * X.shape[0])
    
        return P

    def compute_low_dim_probabilities(self, Y):
        dists = self.pairwise_distances(Y)
        inv_distances = 1. / (1. + dists)
        np.fill_diagonal(inv_distances, 0.)
        Q = inv_distances / np.sum(inv_distances)
        return Q
    

    def momentum(self, t):
        return 0.5 if t < 250 else 0.8



    def run_tsne(self, X, ydim=2, T=1000, l=500, perp=30):
        N = X.shape[0]
        P = self.p_joint(X, perp)

        Y = np.random.normal(loc=0.0, scale=1e-4, size=(N, ydim))
        Y_prev = Y.copy()

        for t in range(T):
            Q = self.compute_low_dim_probabilities(Y)
            grad = self.gradient_descent(P, Q, Y)
            if t % 10 == 0:
                Q = np.maximum(Q, 1e-12)

            Y_new = Y - l * grad + self.momentum(t) * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new

        return Y

    def gradient_descent(self, P, Q, Y):
        pq_diff = P - Q
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        dists = self.pairwise_distances(Y)
        aux = 1 / (1 + dists)
        grad = 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux, 2)).sum(1)
        
        return grad
    