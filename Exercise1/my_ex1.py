import sys, os
import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = df[:,0], df[:,1]
m = np.size(y)

plt.figure()
plt.plot(X, y, 'bx')
plt.show()

# add ones in the first column
X = np.stack((np.ones(m), X), axis=1)

theta = np.zeros(2)
alpha = 0.005

def cost(x, y, m, theta):
    J = 0.5 * np.mean((x @ theta - y)**2)
    return J

def gradient_descent(x, y, m, theta, alpha):
    t0 = theta[0] - alpha * (np.mean(x @ theta - y))
    t1 = theta[1] - alpha * (np.mean((x @ theta - y) * x[:,1]))
    return np.array([t0, t1])

cost(X, y, m, theta)

for alpha in [0.023, 0.024, 0.025, 0.026, 0.027]:
    X, y = df[:,0], df[:,1]
    X = np.hstack((np.ones(m)[:,np.newaxis], X[:,np.newaxis]))
    theta = np.zeros(2)
    for i in range(1000):
        theta = gradient_descent(X, y, m, theta, alpha)
    print(f'alpha: {alpha}, cost: {cost(X, y, m, theta)}, theta: {theta}')
