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
X = np.hstack((np.ones(m)[:,np.newaxis], X[:,np.newaxis]))

theta = np.zeros(2)
alpha = 0.01

def cost(x, y, m, theta):
    J = np.sum((x @ theta - y)**2) / (2*m)
    return J

def gradient_descent(x, y, m, theta, alpha):
    t0 = theta[0] - alpha * (np.mean(x @ theta - y))
    t1 = theta[1] - alpha * (np.mean((x @ theta - y) * x[:,1]))
    return np.array([t0, t1])

cost(X, y, m, theta)

for alpha in [0.001, 0.005, 0.01, 0.02, 0.03]:
    X, y = df[:,0], df[:,1]
    X = np.hstack((np.ones(m)[:,np.newaxis], X[:,np.newaxis]))
    theta = np.zeros(2)
    for i in range(1000):
        theta = gradient_descent(X, y, m, theta, alpha)
    print(f'alpha: {alpha}, cost: {cost(X, y, m, theta)}, theta: {theta}')
