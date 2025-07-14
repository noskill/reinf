import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super(BaseNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class JointNet(BaseNet):
    def __init__(self, x_dim, z_dim, hidden_dims=(128, 64)):
        super(JointNet, self).__init__(x_dim + z_dim, hidden_dims)

    def forward(self, x, z):
        combined = torch.cat([x, z], dim=1)
        return super(JointNet, self).forward(combined)

class MarginalNet(BaseNet):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super(MarginalNet, self).__init__(input_dim, hidden_dims)

class TNetwork(nn.Module):
    """
    T function for MINE: T(x, z) = log t1(x, z) - log t2(x) - log t3(z)
    where t1, t2, t3 are neural net parameterized distributions.
    """
    def __init__(self, x_dim, z_dim, hidden_dims=(128, 64)):
        super(TNetwork, self).__init__()
        # Joint and marginal networks
        self.t1 = JointNet(x_dim, z_dim, hidden_dims)
        self.t2 = MarginalNet(x_dim, hidden_dims)
        self.t3 = MarginalNet(z_dim, hidden_dims)
        # Learnable constant offset added to the T function output
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):
        # Compute factorized log-densities
        log_t1 = self.t1(x, z)
        log_t2 = self.t2(x)
        log_t3 = self.t3(z)
        # Return T(x,z) = log t1(x,z) - log t2(x) - log t3(z) + offset
        return log_t1 - log_t2 - log_t3 + self.offset