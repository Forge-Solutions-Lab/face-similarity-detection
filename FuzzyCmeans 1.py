import numpy as np
import matplotlib.pyplot as plt

X = np.vstack((np.random.randn(15, 2),
               np.random.randn(15, 2)+np.array([3.8, 3.8]),
               np.random.randn(15, 2)+np.array([-3.8, -3.8]),))

def plot(C):
    plt.clf()
    plt.plot(X[:15,0],X[:15,1], '.r')
    plt.plot(X[15:30,0],X[15:30,1], '.g')
    plt.plot(X[30:,0],X[30:,1], '.b')
    plt.plot(C[:,0],C[:,1], 'ok')
    plt.pause(.5)

K = 3
m = 2
C = np.zeros((K, X.shape[1]))

mu = np.random.rand(len(X), K)
while True:
    mu /= np.sum(mu, axis=1)[:, None]
    Cold = C.copy()
    for k in range(K):
        C[k] = np.dot(mu[:, k] ** m, X) / np.sum(mu[:, k] ** m)
        D = np.sqrt(np.sum((X - C[k])**2, axis=1))   # Euclidean distance
        D = D ** (-2/(m-1))
        mu[:, k] = D
    plot(C)
    if np.mean(np.abs(Cold-C)) < 1e-6:
        print('finished')
        break