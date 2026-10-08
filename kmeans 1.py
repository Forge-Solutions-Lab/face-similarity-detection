import numpy as np
import matplotlib.pyplot as plt

X = np.vstack((np.random.randn(20,2),
               np.random.randn(20,2)+np.array([3.8,3.8]),
               np.random.randn(20,2)+np.array([-3.8,-3.8])))
def plot(C):
    plt.clf()
    plt.plot(X[:20,0],X[:20,1],'.r')
    plt.plot(X[20:40,0],X[20:40,1],'.g')
    plt.plot(X[40:,0],X[40:,1],'.b')
    plt.plot(C[:,0],C[:,1],'sk')
    plt.pause(1)

K = 3
C = X[np.random.permutation(len(X))[:K]]
while True:
    plot(C)
    D = np.zeros((len(X),K))
    for k in range(K):
        D[:, k] = np.sum(np.abs(X - C[k]), axis=1)
    idx = np.argmin(D, axis=1)
    Cold = C.copy()
    for k in range(K):
        C[k] = np.mean(X[idx==k])
    if np.sum(np.abs(Cold - C)) == 0:
        print('finished')
        break