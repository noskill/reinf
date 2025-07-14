import torch
import numpy as np
import argparse
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

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


class LogMeanExp(torch.autograd.Function):
    """Custom function for computing log(mean(exp(x))) with corrected gradients
       that replace the empirical mean with a moving average (ma_et) for the backward pass.
    """
    @staticmethod
    def forward(ctx, x, ma_et):
        # Compute the standard mean-exp and its log.
        exp_x = torch.exp(x)
        mean_exp = torch.mean(exp_x)
        result = torch.log(mean_exp)
        # Save tensors required in backward (ma_et, although a constant wrt theta, is needed for gradient scaling).
        ctx.save_for_backward(exp_x, mean_exp, ma_et)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        exp_x, mean_exp, ma_et = ctx.saved_tensors
        # Get the batch size (number of elements along dim=0).
        N = exp_x.shape[0]
        # Normally, d/dx log(mean(exp(x))) = exp_x / (N * mean_exp).
        # Here we use moving average ma_et in place of mean_exp.
        grad_x = grad_output * (exp_x / (N * ma_et))
        # We don't backpropagate through the second argument (ma_et), so return None.
        return grad_x, None


def log_mean_exp_with_moving_avg(x, ma_et):
    return LogMeanExp.apply(x, ma_et)

# Use TNetwork for the T function representation
# Use TNetwork for the T function representation
from t_networks import TNetwork
## -------------------------------------------------------------------
## Define both original and factored T-network implementations
## -------------------------------------------------------------------
## Original T-network: full MLP over joint inputs (x,y)
class OriginalNet(nn.Module):
    """Original MLP-based T-network: single network on (x,y)."""
    def __init__(self, input_size=2):
        super(OriginalNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.ma_et = 1.0
        self.alpha = 0.01

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        x = F.relu(self.layer1(combined))
        x = F.relu(self.layer2(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def mutual_information(self, joint, marginal):
        t_joint = self.forward(joint[0], joint[1])
        t_marginal = self.forward(marginal[0], marginal[1])
        exp_t = torch.exp(t_marginal)
        current = torch.mean(exp_t).item()
        self.ma_et = (1 - self.alpha) * self.ma_et + self.alpha * current
        ma_tensor = torch.tensor(self.ma_et, device=t_marginal.device, dtype=t_marginal.dtype)
        return torch.mean(t_joint) - log_mean_exp_with_moving_avg(t_marginal, ma_tensor)

## Factored T-network: log t1(x,z) - log t2(x) - log t3(z)
class FactoredNet(TNetwork):
    """Factored T-network: log t1(x,z) - log t2(x) - log t3(z)."""
    def __init__(self, input_size=2):
        x_dim = input_size // 2
        z_dim = input_size - x_dim
        super(FactoredNet, self).__init__(x_dim, z_dim)
        self.ma_et = 1.0
        self.alpha = 0.01

    def mutual_information(self, joint, marginal):
        t_joint = self.forward(joint[0], joint[1])
        t_marginal = self.forward(marginal[0], marginal[1])
        exp_t_marginal = torch.exp(t_marginal)
        current_estimate = torch.mean(exp_t_marginal).item()
        self.ma_et = (1 - self.alpha) * self.ma_et + self.alpha * current_estimate
        ma_tensor = torch.tensor(self.ma_et, device=t_marginal.device, dtype=t_marginal.dtype)
        mi = torch.mean(t_joint) - log_mean_exp_with_moving_avg(t_marginal, ma_tensor)
        return mi

# Default alias to the factored implementation
Net = FactoredNet

def build_net(input_size, model_type='factored'):
    """Factory for selecting T-network implementation: 'original' or 'factored'."""
    if model_type == 'original':
        return OriginalNet(input_size)
    elif model_type in ('factored', 'new'):
        return FactoredNet(input_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (batch_x, batch_y, batch_y_shuffle) in enumerate(train_loader):
        batch_x, batch_y, batch_y_shuffle = batch_x.to(device), batch_y.to(device), batch_y_shuffle.to(device)
        joint = (batch_x, batch_y)
        marginal = (batch_x, batch_y_shuffle)

        optimizer.zero_grad()
        # negate to maximize MI
        loss = - model.mutual_information(joint, marginal)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(batch_x)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MINE Implementation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--model-type', choices=['original','factored'], default='factored',
                        help='Which T-network implementation to use')
    args = parser.parse_args()

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate data
    rho = 0.35
    n_samples = 10000
    x, y, true_mi = generate_nonlinear_data(n_samples)
    x, y, true_mi = generate_multivariate_gaussian(n_samples, 10, rho)
    x, y, true_mi = generate_correlated_gaussian(rho, n_samples)

    # Create dataset and deterministic train/test split
    dataset = MINEDataset(x, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    split_gen = torch.Generator()
    split_gen.manual_seed(args.seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=split_gen)

    # Create data loaders with fixed shuffle seed
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=loader_gen)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(args.seed))

    # Initialize model and optimizer
    model = build_net(x.shape[1] + y.shape[1], args.model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    print(f"True MI: {true_mi}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if epoch % 10 == 0 or epoch == args.epochs:
            test_estimates = []
            model.eval()
            with torch.no_grad():
                # Get all data from the test dataset
                batch_x, batch_y, batch_y_shuffle = test_dataset[:]
                batch_x, batch_y, batch_y_shuffle = batch_x.to(device), batch_y.to(device), batch_y_shuffle.to(device)

                joint_test = (batch_x, batch_y)
                marginal_test = (batch_x, batch_y_shuffle)

                # The function returns the loss (negative MI), so we negate it.
                mi_estimate = model.mutual_information(joint_test, marginal_test).item()
                test_estimates.append(mi_estimate)
            avg_test_mi = sum(test_estimates) / len(test_estimates)
            print(f'\nTest MI Estimate: {avg_test_mi:.4f}\n')
            model.train()

        scheduler.step()
        print(f"True MI: {true_mi}")

if __name__ == '__main__':
    main()
