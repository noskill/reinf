#!/usr/bin/env python3
"""
Train and visualize T function of MINE on a bivariate Gaussian distribution.

This script trains the neural network T(x,y) defined in mine.py to estimate mutual
information on 2D correlated Gaussian data. After training, it visualizes the learned
T function over a 2D grid.
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from mine import generate_correlated_gaussian, MINEDataset, build_net
from mine import generate_nonlinear_2d_data


def visualize_learned_function(model, device, noise_std=0.0):
    """
    Visualizes the output of the learned T function over the 2D input space of x.

	1. Create a 2D Grid: Make a grid of points for the x variable's components, x1 and x2.
	2. Calculate y: For each (x1, x2) point on the grid, calculate the corresponding y values using the known function (y1 = x1^2, y2 = x2^2).
	3. Feed Forward:
    """
    print("Visualizing learned T(x,y) function...")
    
    # Ensure the model is in evaluation mode
    model.eval()

    # 1. Create a 2D grid for the x components
    n_points = 200
    x1_range = np.linspace(-3, 3, n_points)
    x2_range = np.linspace(-3, 3, n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # 2. Prepare the x and y tensors
    # x is the grid itself
    x_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float32)

    # y is the function of x
    noise = torch.randn_like(x_tensor) * noise_std
    y_tensor = torch.square(x_tensor) + noise
    
    # We need to pass the tensors to the model in batches to avoid memory issues
    batch_size = 1024
    t_values = []

    with torch.no_grad(): # No need to calculate gradients
        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor[i:i+batch_size].to(device)
            y_batch = y_tensor[i:i+batch_size].to(device)
            
            # 3. Get the output of the T function from your model's forward pass
            t_batch = model(x_batch, y_batch)
            t_values.append(t_batch.cpu())

    # Concatenate results from all batches
    T = torch.cat(t_values, dim=0).numpy()

    # 4. Reshape for plotting
    T_grid = T.reshape(n_points, n_points)

    # 5. Plot the result
    plt.figure(figsize=(8, 7))
    contour = plt.contourf(X1, X2, T_grid, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Learned T(x, y) value')
    plt.title('Visualization of the Learned T Function', fontsize=14)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show(block=False)
    plt.pause(0.1)
    return
  
def visualize_marginals(model, x, y, args, device):
    """
    Visualize marginal networks t2(x) and t3(z) for factored T.
    Supports both 1D (line) and 2D (contour) inputs.
    """
    x_dim = x.shape[1]
    z_dim = y.shape[1]
    # t2 over x
    if x_dim == 1:
        xs = np.linspace(args.xlim[0], args.xlim[1], args.grid_size)
        xg = torch.from_numpy(xs.reshape(-1,1)).float().to(device)
        with torch.no_grad():
            t2v = model.t2(xg).cpu().numpy().squeeze()
        plt.figure(figsize=(6,4))
        plt.plot(xs, t2v)
        plt.title('t2(x)')
        plt.xlabel('x')
        plt.ylabel('t2(x)')
        plt.grid(True)
    else:
        xs = np.linspace(args.xlim[0], args.xlim[1], args.grid_size)
        ys = np.linspace(args.ylim[0], args.ylim[1], args.grid_size)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        xg = torch.from_numpy(grid).float().to(device)
        with torch.no_grad():
            t2v = model.t2(xg).cpu().numpy().reshape(args.grid_size, args.grid_size)
        plt.figure(figsize=(6,5))
        plt.contourf(xx, yy, t2v, levels=50, cmap='plasma')
        plt.title('t2(x)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
    # t3 over z
    if z_dim == 1:
        zs = np.linspace(args.ylim[0], args.ylim[1], args.grid_size)
        zg = torch.from_numpy(zs.reshape(-1,1)).float().to(device)
        with torch.no_grad():
            t3v = model.t3(zg).cpu().numpy().squeeze()
        plt.figure(figsize=(6,4))
        plt.plot(zs, t3v)
        plt.title('t3(z)')
        plt.xlabel('z')
        plt.ylabel('t3(z)')
        plt.grid(True)
    else:
        xs = np.linspace(args.xlim[0], args.xlim[1], args.grid_size)
        ys = np.linspace(args.ylim[0], args.ylim[1], args.grid_size)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        zg = torch.from_numpy(grid).float().to(device)
        with torch.no_grad():
            t3v = model.t3(zg).cpu().numpy().reshape(args.grid_size, args.grid_size)
        plt.figure(figsize=(6,5))
        plt.contourf(xx, yy, t3v, levels=50, cmap='plasma')
        plt.title('t3(z)')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.colorbar()
    return
  
def visualize_data_nonlinear2d(x, y):
    """
    Visualize source and transformed 2D nonlinear data distributions.
    """
    # Convert to numpy
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    # Source distribution
    plt.figure(figsize=(6, 5))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=10, alpha=0.5)
    plt.title('Source 2D Gaussian Distribution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.5)
    # Transformed distribution
    plt.figure(figsize=(6, 5))
    plt.scatter(y_np[:, 0], y_np[:, 1], s=10, alpha=0.5)
    plt.title('Transformed Nonlinear Distribution (y = x^2 + noise)')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show(block=False)
    plt.pause(0.1)
    return


def train_T(model, loader, lr, epochs, device):
    """
    Train the T network on provided data loader.

    Args:
        model: T-network model
        loader: DataLoader yielding (x, y, y_shuffle) batches
        lr: learning rate
        epochs: number of training epochs
        device: torch device

    Returns:
        Trained model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y, batch_y_shuffle in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_shuffle = batch_y_shuffle.to(device)

            joint = (batch_x, batch_y)
            marginal = (batch_x, batch_y_shuffle)

            optimizer.zero_grad()
            loss = -model.mutual_information(joint, marginal)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs}, loss={avg_loss:.4f}")

    return model


def visualize_T(model, rho, true_mi, device, grid_size, xlim, ylim):
    """
    Visualize the learned T function over a grid of (x, y) values.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_tensor = torch.from_numpy(grid).float().to(device)
    x_grid = grid_tensor[:, 0:1]
    y_grid = grid_tensor[:, 1:2]

    with torch.no_grad():
        t_vals = model.forward(x_grid, y_grid).cpu().numpy().reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, t_vals, levels=50, cmap="viridis")
    plt.colorbar(label="T(x, y)")
    plt.title(f"T function after training (rho={rho:.2f}, true MI={true_mi:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train and visualize T function of MINE on 2D correlated Gaussian data"
    )
    parser.add_argument("--rho", type=float, default=0.8, help="Correlation coefficient")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--grid-size", type=int, default=100, help="Grid size for visualization"
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        help="Limits for x-axis in visualization",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        help="Limits for y-axis in visualization",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for training and inference"
    )
    parser.add_argument(
        "--model-type", choices=["original","factored"], default="factored",
        help="Which T-network implementation to use"
    )
    parser.add_argument(
        "--data", choices=["correlated","nonlinear2d"], default="correlated",
        help="Which data to generate: 'correlated' for Gaussian, 'nonlinear2d' for 2D nonlinear"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(args.device)
    # Generate and prepare data based on selected function
    if args.data == "correlated":
        x, y, true_mi = generate_correlated_gaussian(args.rho, args.n_samples)
    elif args.data == "nonlinear2d":
        # generate_nonlinear_2d_data(n_samples, rho)
        x, y, true_mi = generate_nonlinear_2d_data(args.n_samples, rho=args.rho)
    else:
        raise ValueError(f"Unknown data type: {args.data}")
    # x,y are [n, d]; create dataset and loader
    dataset = MINEDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Build and train model
    input_size = x.shape[1] + y.shape[1]
    model = build_net(input_size=input_size, model_type=args.model_type).to(device)
    model = train_T(model, loader, args.lr, args.epochs, device)
    # Visualization
    if args.data == "correlated":
        visualize_T(model, args.rho, true_mi, device, args.grid_size, args.xlim, args.ylim)
    elif args.data == "nonlinear2d":
        visualize_data_nonlinear2d(x, y)
        visualize_learned_function(model, device)
    else:
        print(f"No visualization implemented for data='{args.data}'")
    # If factored model, visualize marginals t2 and t3
    if args.model_type == 'factored':
        visualize_marginals(model, x, y, args, device)
    # Block until all figures are closed
    plt.show()


if __name__ == "__main__":
    main()
