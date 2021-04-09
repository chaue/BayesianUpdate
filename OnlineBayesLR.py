# Bayesian Regression

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# generate toy data
dim = 1
numpoints = 100
x = np.linspace(-3, 3, numpoints)
x = x[np.newaxis].T

slope = 3
eps_mu = 0
beta = 0.25
eps = np.random.normal(eps_mu, 1/beta, size = (numpoints, dim))
t = slope * x + eps

# phi is a column of ones next to the column of x's
phi = np.hstack((np.ones((numpoints, 1)), x))

# choose an uninformative prior for simplicity
m_0 = np.array([[0], [0]])
S_0 = np.eye(2)
beta = 0.25

S_0inv = np.linalg.lstsq(S_0, np.eye(2), rcond = None)[0]
S_n = np.linalg.lstsq(S_0inv + beta * phi.T @ phi, np.eye(2), rcond = None)[0]
m_n = S_n @ (S_0inv @ m_0 + beta * phi.T @ t)

line_x_b = np.linspace(np.min(x)-5, np.max(x) + 5, 100)
line_y_b = m_n[0] + m_n[1]*line_x_b

plt.scatter(x, t)
plt.plot(line_x_b, line_y_b, 'r')
plt.show()

# Online Updates

# we will loop through phi to mimic an incoming stream of data
# same priors as before
m_n = np.array([[0.0], [1.0]])
S_n = np.eye(2)
S_ninv = np.linalg.lstsq(S_n, np.eye(2), rcond = None)[0]
beta = 0.25

for i in range(len(phi[:, 0])):
    S_ninv_old = S_ninv
    
    S_ninv = S_ninv + beta * phi[np.newaxis, i, :].T @ phi[np.newaxis, i, :]
    S_n = np.linalg.lstsq(S_ninv, np.eye(2), rcond = None)[0]
    m_n = S_n @ (S_ninv_old @ m_n + beta * phi[np.newaxis, i, :].T * t[i])
	
line_x_on = np.linspace(np.min(x)-5, np.max(x) + 5, 100)
line_y_on = m_n[0] + m_n[1]*line_x_on

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].title.set_text("Bayesian")
ax[0].scatter(x, t)
ax[0].plot(line_x_b, line_y_b, 'r')
ax[1].title.set_text("Online Bayesian")
ax[1].scatter(x, t)
ax[1].plot(line_x_on, line_y_on, 'r')
plt.show()