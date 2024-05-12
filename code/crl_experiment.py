import numpy as np
import matplotlib.pyplot as plt

mu = 5.0
sigma = 100.0
blk_size = 16
n_blocks = 2
in_size = 16
out_size = 8

# Initialize function
def _generate_random_weights(widths):
    global mu, sigma
    weights = []
    for i in range(1, len(widths)):
        A = np.random.normal(loc=mu, scale=sigma, size=(widths[i], widths[i-1]))
        weights.append(A)
    return weights

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

if __name__ =='__main__':
    SCs = []
    FNs = []
    num_blocks = list(range(2, 50+1))
    for n_blocks in num_blocks:
        # Initialize network
        widths = [in_size] + [blk_size]*n_blocks + [out_size]
        weights = _generate_random_weights(widths)
        L = len(weights)
        
        # Calculate spectral complexity
        spectral_complexity = 0.0
        frobenius_norms = 1.0
        for l in range(L):
            spectral_complexity += (
                _l21_norm(weights[l]) / _spectral_norm(weights[l])
            ) ** (2/3)
            frobenius_norms *= _frobenius_norm(weights[l])
        spectral_complexity = (spectral_complexity + out_size**(1/3)) ** (3/2)
        spectral_complexity = spectral_complexity * np.prod([_spectral_norm(weights[l]) for l in range(L)])
        frobenius_norms *= np.sqrt(out_size*(n_blocks+2))
    
        print(spectral_complexity, frobenius_norms)
        SCs.append(spectral_complexity)
        FNs.append(frobenius_norms)
        
    plt.plot(num_blocks, np.log(FNs), label='Product of Frobenius norm', marker='o')
    plt.plot(num_blocks, np.log(SCs), label='Spectral complexity', marker='v')
    plt.legend()
    plt.show()