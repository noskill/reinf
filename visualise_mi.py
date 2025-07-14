import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# 1. Define Distribution Parameters
rho = 0.6  # Correlation coefficient
mu = [0, 0] # Mean
# Covariance matrix for a standard bivariate normal
cov = [[1, rho], 
       [rho, 1]]

# 2. Create a grid of (x, y) points
x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 3. Calculate Probability Density Functions (PDFs)
# Joint PDF: p(x, y)
joint_pdf = multivariate_normal(mu, cov).pdf(pos)

# Marginal PDFs: p(x) and p(y)
# For a standard bivariate normal, the marginals are standard normals
marginal_x_pdf = norm.pdf(X)
marginal_y_pdf = norm.pdf(Y)

# 4. Calculate Pointwise Mutual Information (PMI)
# To avoid log(0) errors, we add a small epsilon
epsilon = 1e-9
pmi = np.log(joint_pdf / (marginal_x_pdf * marginal_y_pdf + epsilon) + epsilon)

# 5. Plot the results
plt.figure(figsize=(8, 7))
contour = plt.contourf(X, Y, pmi, levels=50, cmap='coolwarm')
plt.colorbar(contour, label='Pointwise Mutual Information (PMI)')
plt.title(f'PMI for Correlated Gaussians ($\\rho$ = {rho})', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
