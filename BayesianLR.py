# Frequentist Regression

# simple linear regression with no basis function for simplicity and sake of visualization
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

dim = 1
# generate toy data
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

# pinv is generally avoided since singular matrices can cause problems
#w = np.linalg.pinv(phi.T @ phi) @ phi.T @ t

# use lstsq instead, it solves for x in Ax = b
w = np.linalg.lstsq(phi.T @ phi, phi.T @ t, rcond=None)[0]

# plot data and fitted reg. line
line_x_ls = np.linspace(np.min(x)-5, np.max(x) + 5, 100)
line_y_ls = w[0] + w[1]*line_x_ls
plt.scatter(x, t)
plt.plot(line_x_ls, line_y_ls, 'r')
plt.show()

# Bayesian Regression

# choose an uninformative prior for simplicity
m_0 = np.array([[0], [0]])
S_0 = np.eye(2)
beta = 0.25

S_0inv = np.linalg.lstsq(S_0, np.eye(2), rcond = None)[0]
S_n = np.linalg.lstsq(S_0inv + beta * phi.T @ phi, np.eye(2), rcond = None)[0]
m_n = S_n @ (S_0inv @ m_0 + beta * phi.T @ t)

line_x_b = np.linspace(np.min(x)-5, np.max(x) + 5, 100)
line_y_b = m_n[0] + m_n[1]*line_x_b

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].title.set_text("Least Squares")
ax[0].scatter(x, t)
ax[0].plot(line_x_ls, line_y_ls, 'r')
ax[1].title.set_text("Bayesian")
ax[1].scatter(x, t)
ax[1].plot(line_x_b, line_y_b, 'r')
plt.show()

# Regularization Effects of Bayesian Regression

# zero-mean prior
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# generate toy data
dim = 1
numpoints = 100
x = np.linspace(-3, 3, numpoints)
x = x[np.newaxis].T

eps_mu = 4
beta = 0.125
eps = np.random.normal(eps_mu, 1/beta, size = (dim, numpoints)).T
t = 0.5 * x**4 + eps

# phi is 5 columns of x^i for i in {0, 1,... 4}
phi = np.hstack((np.ones((numpoints, 1)), x, x**2, x**3, x**4))

# use lstsq instead, it solves for x in Ax = b
w = np.linalg.lstsq(phi.T @ phi, phi.T @ t, rcond=None)[0]
print("frequentist:", w.T)
# draw fitted polynomial
line_x_ls = np.linspace(np.min(x), np.max(x), 100)
line_y_ls = w[0] + w[1]*line_x_ls + w[2]*line_x_ls**2 + w[3]*line_x_ls**3 + w[4]*line_x_ls**4 

# zero-mean isotropic prior with alpha = 10
m_0 = np.array([[0], [0], [0], [0], [0]])
S_0 = 0.1 * np.eye(5)
beta = 0.125

# calculate bayesian params
S_0inv = np.linalg.lstsq(S_0, np.eye(5), rcond = None)[0]
S_n = np.linalg.lstsq(S_0inv + beta * phi.T @ phi, np.eye(5), rcond = None)[0]
m_n = S_n @ (S_0inv @ m_0 + beta * phi.T @ t)
# draw bayesian fitted polynomial
line_x_b = np.linspace(np.min(x), np.max(x), 100)
line_y_b = m_n[0] + m_n[1]*line_x_b + m_n[2]*line_x_b**2 + m_n[3]*line_x_b**3 + m_n[4]*line_x_b**4 
print("Bayesian:   ", m_n.T)

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].title.set_text("Least Squares")
ax[0].scatter(x, t)
ax[0].plot(line_x_ls, line_y_ls, 'r')
ax[1].title.set_text("Bayesian")
ax[1].scatter(x, t)
ax[1].plot(line_x_b, line_y_b, 'r')
plt.show()

# non-zero-mean prior

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# generate toy data
numpoints = 20
x = np.linspace(-3, 3, numpoints)
x = x[np.newaxis].T

eps_mu = 4
beta = 0.125
eps = np.random.normal(eps_mu, 1/beta, size = (dim, numpoints)).T
t = 0.5 * x**4 + eps

# phi is 5 columns of x^i for i in {0, 1,... 4}
phi = np.hstack((np.ones((numpoints, 1)), x, x**2, x**3, x**4))

# use lstsq instead, it solves for x in Ax = b
w = np.linalg.lstsq(phi.T @ phi, phi.T @ t, rcond=None)[0]
print("frequentist:        ", w.T)
# draw fitted polynomial
line_x_ls = np.linspace(np.min(x), np.max(x), 100)
line_y_ls = w[0] + w[1]*line_x_ls + w[2]*line_x_ls**2 + w[3]*line_x_ls**3 + w[4]*line_x_ls**4 

# zero-mean isotropic prior with alpha = 10
m_0 = np.array([[0], [0], [0], [0], [0]])
S_0 = 0.1 * np.eye(5)
beta = 0.125

# calculate bayesian params
S_0inv = np.linalg.lstsq(S_0, np.eye(5), rcond = None)[0]
S_n = np.linalg.lstsq(S_0inv + beta * phi.T @ phi, np.eye(5), rcond = None)[0]
m_n = S_n @ (S_0inv @ m_0 + beta * phi.T @ t)
# draw bayesian fitted polynomial
line_x_b1 = np.linspace(np.min(x), np.max(x), 100)
line_y_b1 = m_n[0] + m_n[1]*line_x_b + m_n[2]*line_x_b**2 + m_n[3]*line_x_b**3 + m_n[4]*line_x_b**4 
print("Bayesian:           ", m_n.T)

# nonzero-mean isotropic prior with alpha = 10
m_0 = np.array([[0], [0], [10], [0], [0]])
S_0 = 0.1 * np.eye(5)
beta = 0.125

# calculate bayesian params
S_0inv = np.linalg.lstsq(S_0, np.eye(5), rcond = None)[0]
S_n = np.linalg.lstsq(S_0inv + beta * phi.T @ phi, np.eye(5), rcond = None)[0]
m_n = S_n @ (S_0inv @ m_0 + beta * phi.T @ t)
# draw bayesian fitted polynomial
line_x_b2 = np.linspace(np.min(x), np.max(x), 100)
line_y_b2 = m_n[0] + m_n[1]*line_x_b + m_n[2]*line_x_b**2 + m_n[3]*line_x_b**3 + m_n[4]*line_x_b**4 
print("nonzero Bayesian:   ", m_n.T)

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].title.set_text("Least Squares")
ax[0].scatter(x, t)
ax[0].plot(line_x_ls, line_y_ls, 'r')
ax[1].title.set_text("Bayesian")
ax[1].scatter(x, t)
ax[1].plot(line_x_b1, line_y_b1, 'r')
ax[2].title.set_text("nonzero Bayesian")
ax[2].scatter(x, t)
ax[2].plot(line_x_b2, line_y_b2, 'r')
plt.show()