import torch
import numpy as np
import random


def generate_correlated_gaussian(rho, n_samples):
    """
    Generate correlated Gaussian samples with correlation coefficient rho
    """
    # Draw standard normal samples as column vectors
    x = torch.randn(n_samples, 1)
    z = torch.randn(n_samples, 1)
    # Convert rho to tensor to avoid type mismatch
    rho_tensor = torch.tensor(rho)
    y = rho_tensor * x + torch.sqrt(1 - rho_tensor**2) * z
    true_mi = -0.5 * np.log(1 - rho**2)
    return x, y, true_mi


def generate_multivariate_gaussian(n_samples, dim, rho):
    """
    Generates correlated multivariate Gaussian data.
    X and Y are `dim`-dimensional vectors.
    """
    # Create a covariance matrix for the 2*dim vector [X, Y]
    # Cov(X) = Cov(Y) = I
    # Cov(X, Y) = rho * I
    mu = np.zeros(2 * dim)
    cov_xx = np.eye(dim)
    cov_yy = np.eye(dim)
    cov_xy = rho * np.eye(dim)

    # Build the full covariance matrix
    cov = np.block([[cov_xx, cov_xy],
                    [cov_xy.T, cov_yy]])

    # Generate samples
    data = np.random.multivariate_normal(mu, cov, n_samples)
    x = torch.tensor(data[:, :dim], dtype=torch.float32)
    y = torch.tensor(data[:, dim:], dtype=torch.float32)

    # True MI for multivariate Gaussian
    # I(X;Y) = -0.5 * log(det(I - rho^2)) = -0.5 * log((1-rho^2)^dim)
    true_mi = -0.5 * np.log(np.linalg.det(cov_xx @ cov_yy - cov_xy @ cov_xy.T))
    # A simpler form for this specific covariance structure:
    true_mi_simple = -0.5 * dim * np.log(1 - rho**2)


    # Note: Your network input_size must be changed to 2 * dim
    # self.layer1 = nn.Linear(2 * dim, 128)
    # And the concatenation logic needs to handle matrices:
    # combined = torch.cat([x, y], dim=1)

    return x, y, true_mi


def generate_nonlinear_2d_data(n_samples, rho=0.6, noise_std=0.1):
    """
    Generates data where a 2D Gaussian is passed through a non-linear function.

    1. Starts with a 2D correlated Gaussian variable `x`.
    2. Creates a second variable `y` by element-wise squaring `x` and adding noise.
    """
    # Define the initial 2D Gaussian distribution
    dim = 2
    mu = np.zeros(dim)
    cov = [[1, rho],
           [rho, 1]]

    # Generate the first variable, x
    x_data = np.random.multivariate_normal(mu, cov, n_samples)
    x = torch.tensor(x_data, dtype=torch.float32)

    # Generate the second variable, y = x^2 + noise
    noise = torch.randn(n_samples, dim) * noise_std
    y = torch.square(x) + noise

    # True MI is not known analytically, but it should be significantly positive
    true_mi_label = "Unknown (but > 0)"

    return x, y, true_mi_label


def generate_independent_data(n_samples):
    """Generates two independent standard normal vectors."""
    x = torch.randn(n_samples)
    y = torch.randn(n_samples)
    true_mi = 0.0
    return x, y, true_mi


def generate_nonlinear_data(n_samples):
    """
    Generates data with a non-linear (sine wave) relationship.
    The ground truth MI is not easily known, but should be > 0.
    """
    x = torch.rand(n_samples) * 4 * np.pi - 2 * np.pi # X from -2pi to 2pi
    noise = torch.randn(n_samples) * 0.2
    y = torch.sin(x) + noise
    # True MI is unknown, but we expect a positive value.
    # We can't calculate 'true_mi' here, so we just test for a positive result.
    return x, y, "Unknown (but > 0)"


class MINEDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.float()  # Ensure float type
        self.y = y.float()
        self.length = len(x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Handle both single indices and lists/tensors of indices
        """
        # Convert idx to tensor if it's a list
        if isinstance(idx, list):
            idx = torch.tensor(idx)

        # Handle both single items and batches
        if torch.is_tensor(idx):
            x_i = self.x[idx]
            y_i = self.y[idx]

            # Generate random indices for shuffled y, same size as input idx
            y_shuffle_idx = torch.randint(0, self.length, (len(idx),))
            y_shuffle = self.y[y_shuffle_idx]
        else:
            # Single item case
            x_i = self.x[idx]
            y_i = self.y[idx]

            # Generate single random index for shuffled y
            y_shuffle_idx = torch.randint(0, self.length, (1,)).item()
            y_shuffle = self.y[y_shuffle_idx]
        return x_i, y_i, y_shuffle


