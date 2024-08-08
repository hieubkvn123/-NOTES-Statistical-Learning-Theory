# Import
import numpy as np
import matplotlib.pyplot as plt

# Initialize matrix
mu = 0.0
sigma = 2.0
n_sim = 100
N = 32

# Some utility
def _l21_norm(A):
    norm = 0.0
    for j in range(A.shape[1]):
        aj_l2 = np.linalg.norm(A[:, j])
        norm += aj_l2
    return norm

def _frobenius_norm(A):
    return np.linalg.norm(A, ord='fro')

def _spectral_norm(A):
    sv = np.linalg.svd(A).S
    return max(sv)

if __name__ == '__main__':    
    S_mu, S_std = [], []
    L_mu, L_std = [], []
    F_mu, F_std = [], []
    sizes = list(range(2, 50))
    for d in sizes:
        print(d)
        S_tmp, L_tmp, F_tmp = [], [], []
        
        for i in range(n_sim):
            A = np.random.normal(loc=mu, scale=sigma, size=(d, N))
            spectral_norm = _spectral_norm(A)
            l21_norm = _l21_norm(A.T)
            fro_norm = _frobenius_norm(A)
            
            S_tmp.append(spectral_norm)
            L_tmp.append(l21_norm)
            F_tmp.append(fro_norm)
        
        S_mu.append(np.mean(S_tmp))
        S_std.append(np.std(S_tmp))
        L_mu.append(np.mean(L_tmp))
        L_std.append(np.std(L_tmp))
        F_mu.append(np.mean(F_tmp))
        F_std.append(np.std(F_tmp))
    
    plt.figure(figsize=(30, 20))
    plt.errorbar(sizes, S_mu, S_std, label='Spectral norm', marker='o')
    plt.errorbar(sizes, L_mu, L_std, label='L21 norm', marker='o')
    plt.errorbar(sizes, F_mu, F_std, label='Frobenius norm', marker='o')
    plt.plot(sizes, np.array(S_mu), label='Spectral norm times N', marker='v')
    plt.plot(sizes, np.array(F_mu) * np.sqrt(d), label='Frobenius norm times d**0.5', marker='v')
    plt.xlabel('Matrix row size (d)')
    plt.ylabel('Matrix norm')
    plt.legend()
    
    plt.show()
        
