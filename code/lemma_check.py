import numpy as np

# Initialize matrix
N, d = 100, 32
A = np.random.normal(size=(N, d))
U = A.T @ A

# Eigen decomposition
result = np.linalg.eig(U)
lambda_max = np.max(result.eigenvalues)
print(lambda_max)

# Initialize vectors
success = True
for i in range(100000):
    x = np.random.normal(loc=0, scale=1.0, size=(d,))
    y = np.random.normal(loc=0, scale=1.0, size=(d,))
    if(lambda_max * np.abs(np.inner(x, y)) - np.inner(x, U@y) <= 0):
            success = False
            
print(success)