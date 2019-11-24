import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from generalized_elastic_net import GeneralizedElasticNetSolver

## generating symmetric signal

# signal length
N = 4096
# number of spikes in the signal
T = 160
# number of observations to make
K = 1024
random.seed(0)
# random +/- 1 signal
x = np.zeros((N, 1))
q = np.random.permutation(N)
x[q[0:T]] = np.sign(np.random.randn(T, 1))

# measurement matrix
A = np.random.randn(K, N)
A = scipy.linalg.orth(A.transpose()).transpose()

# observations
y = np.reshape(A.dot(x), (A.dot(x).shape[0],)) + np.random.normal(0, 0.1, K)

## solve the en problem
s = GeneralizedElasticNetSolver()

# given all parameters
lam_1 = 10
lam_2 = 0
p = A.shape[1]
Pmat = np.diag([1] * N)
dvec = np.ones(p) * T / p
dvec[q[0:T]] = 0
lowbo = -1 * np.ones(N)
upbo = 1 * np.ones(N)
coeffs = s.solve(A, y, lam_1, lam_2, lowbo, upbo, dvec, Pmat, err_tol=1e-8, text='On', text_fr=1000)

errors = np.concatenate(x) - coeffs
plt.rcParams["figure.figsize"] = (7, 1)
plt.plot(x, 'steelblue')
plt.title('True signal')
plt.show()

plt.rcParams["figure.figsize"] = (7, 1)
plt.plot(coeffs, "steelblue")
plt.title('Recovery signal ')
plt.show()

plt.rcParams["figure.figsize"] = (7, 1)
plt.plot(errors, 'sandybrown')
plt.title('Recovery errors ')
plt.ylim([-1, 1])
plt.show()
print(np.sum((coeffs.reshape(x.shape) - x) ** 2) / len(x))
