import torch
import pdb
import math
import numpy as np
import ot  # Install with: pip install POT

from geomloss import SamplesLoss

def fast_wasserstein_geomloss(p, q):
    """
    Computes Wasserstein distance using GeomLoss for fast Sinkhorn approximation.
    """
    loss = SamplesLoss("sinkhorn", p=1, blur=0.05, diameter=0.02)  # `blur` is regularization
    # print(p.flatten(start_dim=1).shape)
    wd = loss(p.flatten(start_dim=1), q.flatten(start_dim=1))
    # print(wd.shape)
    return torch.mean(wd)


def wasserstein_distance_s1(predicted_density, true_density, grid):
    """
    Calculate the approximate Wasserstein distance (Earth Mover's Distance) between two distributions on S1.
    Both predicted_density and true_density should be 1D or 2D torch tensors of the same shape.
    grid should be a 1D torch tensor representing the angular grid coordinates in radians.
    """

    # Ensure densities are probability distributions
    predicted_density = predicted_density / predicted_density.sum(dim=-1, keepdim=True)
    true_density = true_density / true_density.sum(dim=-1, keepdim=True)

    # Handle batch dimension if present
    if predicted_density.ndim == 1:
        predicted_density = predicted_density.unsqueeze(0)
        true_density = true_density.unsqueeze(0)

    batch_size = predicted_density.shape[0]
    grid = grid.unsqueeze(0).repeat(batch_size, 1)  # Repeat grid for batch size

    # Compute pairwise angular distances on S1
    angular_diff = torch.abs(grid.unsqueeze(-1) - grid.unsqueeze(-2))  # Shape: [batch_size, N, N]
    angular_distances = torch.minimum(angular_diff, 2 * torch.pi - angular_diff)  # Shortest path on S1

    # Compute the transport cost
    transport_cost = angular_distances * torch.abs(predicted_density.unsqueeze(-1) - true_density.unsqueeze(-2))

    wasserstein_dist = torch.mean(transport_cost.sum(dim=(-2, -1)))  # Average over batch

    return wasserstein_dist


def wasserstein_distance_s1_simple(predicted_density, true_density, grid):
    """
    Simplified Wasserstein distance for S1 when grid points are aligned.
    """

    # Normalize densities
    predicted_density = predicted_density / predicted_density.sum(dim=-1, keepdim=True)
    true_density = true_density / true_density.sum(dim=-1, keepdim=True)
    batch_size = predicted_density.shape[0]
    grid = grid.unsqueeze(0).repeat(batch_size, 1)
    # Compute angular distance considering S1 topology
    angular_diff = torch.abs(grid - grid.roll(1))  # Grid spacing
    distance = torch.sum(angular_diff * torch.abs(predicted_density - true_density))

    return distance



# def wasserstein_distance_2d(predicted_density, true_density, grid_x, grid_y):
#     """
#     Calculate the approximate Wasserstein distance (Earth Mover's Distance) between two 2D distributions.
#     Both predicted_density and true_density should be 2D torch tensors of the same shape.
#     grid_x and grid_y should be 2D torch tensors representing the grid coordinates.
#     """
#     device = torch.device('cpu')
#     # Normalize densities to sum to 1 (make them probability distributions)
#     predicted_density = (predicted_density / predicted_density.sum(dim=(1,2),keepdim=True)).to(device)
#     true_density = (true_density / true_density.sum(dim=(1,2),keepdim=True)).to(device)

#     # Create a grid of coordinates
#     batch_size = predicted_density.size(0)
#     grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
#     grid_points = torch.tile(grid_points.unsqueeze(0), (batch_size, 1, 1))

#     predicted_density_flat = predicted_density.flatten(start_dim=1)# Flatten the densities
#     true_density_flat = true_density.flatten(start_dim=1)# Flatten the densities
    
#     grid_diff_x = torch.diff(grid_x, prepend=grid_x[..., :1])
#     grid_diff_y = torch.diff(grid_y, prepend=grid_y[..., :1])
#     torch.diff(grid_x,, dim=0)
#     pdb.set_trace()
#     # Compute the Wasserstein distance using a simplified cost function
#     wasserstein_dist = torch.mean( grid_diff_x.unsqueeze(-1) * grid_diff_y.unsqueeze(0) * torch.abs(predicted_density_flat[:, None] - true_density_flat[None, :]))

#     return wasserstein_dist


def wasserstein_dist_2d(predicted_density, true_density, grid_x, grid_y):
# Create 2D meshgrid for positions in R²
    X, Y = torch.meshgrid(grid_x, grid_y, indexing="ij")

    # Flatten grid positions to shape [2500, 2]
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Shape: [2500, 2]

    predicted_density = (predicted_density / predicted_density.sum(dim=(1,2),keepdim=True))
    # .to(predicted_density.device)
    true_density = (true_density / true_density.sum(dim=(1,2),keepdim=True))
    # .to(predicted_density.device)

    # Flatten densities into 1D vectors
    predicted_density_flat = predicted_density.flatten(start_dim=1)
    true_density_flat = true_density.flatten(start_dim=1)

    # Compute cost matrix (Euclidean distance between grid points)
    cost_matrix = torch.cdist(grid_points, grid_points, p=2)  # Shape: [2500, 2500]
    cost_matrix = torch.tile(cost_matrix.unsqueeze(0), (predicted_density.size(0), 1))
    pdb.set_trace()
    # Compute Wasserstein-2 distance using optimal transport
    wasserstein_dist = ot.emd2(predicted_density_flat, true_density_flat, cost_matrix)

    return torch.mean(wasserstein_dist)

def wasserstein_2d(p, q):
    """Computes approximate 2D Wasserstein distance using row & column CDFs."""
    cdf_p_x = torch.cumsum(p, dim=1)  # CDF over x-axis
    cdf_q_x = torch.cumsum(q, dim=1)
    
    cdf_p_y = torch.cumsum(p, dim=2)  # CDF over y-axis
    cdf_q_y = torch.cumsum(q, dim=2)

    wd = torch.sum(torch.abs(cdf_p_x - cdf_q_x), dim=(1, 2)) + torch.sum(torch.abs(cdf_p_y - cdf_q_y), dim=(1, 2))
    return torch.mean(wd)

def wasserstein_distance_2d(predicted_density, true_density):
    grid_size = [50,50]
    batch_size = predicted_density.size(0)
    # Create the coordinate grid
    x_coords, y_coords = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing='ij')
    grid_points = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float().to(predicted_density.device) # (100, 2)

    # Compute the cost matrix (Euclidean distance between grid points)
    C = torch.cdist(grid_points, grid_points, p=2)  # (100, 100)

    predicted_density /= predicted_density.sum(dim=(1, 2), keepdim=True)  # Normalize each PDF

    # # Choose a reference distribution (e.g., the first one)
    # reference_pdf = pdfs[0].flatten()

    # Compute Wasserstein distance for each batch
    wasserstein_distances = torch.zeros(batch_size)

    for i in range(batch_size):
        current_pdf = predicted_density[i].flatten()
        reference_pdf = true_density[i].flatten()
        
        # Solve the linear program using torch (Sinkhorn approximation)
        # Solve using torch.linalg.lstsq() for an exact solution or a differentiable approach
        # print(C.shape, reference_pdf.shape, current_pdf.shape)
        transport_plan= torch.linalg.lstsq(C, reference_pdf - current_pdf).solution
        
        # Compute Wasserstein distance (sum of transport plan * cost)
        wasserstein_distances[i] = (transport_plan.abs() * C).mean()
    
    return wasserstein_distances.mean()

def sinkhorn_knopp(predicted_density, true_density, C, reg=0.05, n_iters=20):
    """
    Computes the Sinkhorn-Knopp approximation of the Wasserstein-1 distance.
    
    Args:
        a (Tensor): Source probability distribution (flattened) (N, 100)
        b (Tensor): Target probability distribution (flattened) (N, 100)
        C (Tensor): Cost matrix (100, 100)
        reg (float): Entropy regularization parameter
        n_iters (int): Number of Sinkhorn iterations

    Returns:
        Tensor: Approximated Wasserstein distances (N,)
    """
    K = torch.exp(-C / reg)  # Gibbs kernel (element-wise exponential)
    temp_1 = torch.ones_like(predicted_density) / predicted_density.sum(dim=-1, keepdim=True)
    # / predicted_density.shape[-1]  # Initial scaling vectors (N, 100)
    temp_2 = torch.ones_like(true_density) / true_density.sum(dim=-1, keepdim=True)
    #  / true_density.shape[-1]

    for iter_ in range(n_iters):
        temp_1 = predicted_density / (temp_2 @ K + 1e-8)  # Row scaling
        temp_2 = true_density / (temp_1 @ K.T + 1e-8)  # Column scaling

    # Compute Sinkhorn distance: sum(u * K * v * C)
    transport_plan = temp_1[:, :, None] * K * temp_2[:, None, :]
    wasserstein_distance = torch.mean(transport_plan * C, dim=(1, 2)) 
    return wasserstein_distance

import torch



def wasserstein_2d(predicted_density, true_density, args):
    # Define grid sizes
    grid_size = args.grid_size[:-1]

    device = torch.device("cpu")

    # Create the coordinate grid
    x_coords, y_coords = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing='ij')
    grid_points = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float().to(device)  # (100, 2)
    # Compute the cost matrix (Euclidean distance between grid points)
    C = torch.cdist(grid_points, grid_points, p=2)  # (100, 100)
    C = C / C.max()

    # predicted_density /= predicted_density.sum(dim=(1, 2), keepdim=True)  # Normalize each PDF

    # Flatten PDFs
    pdfs_flat = predicted_density.reshape(args.batch_size, -1).to(device)  # (batch_size, 100)

    true_density_flat = true_density.reshape(args.batch_size, -1).to(device)  # (batch_size, 100)
    
    # Choose a reference distribution (e.g., the first one)
    # reference_pdf = pdfs_flat[0].unsqueeze(0).repeat(batch_size, 1)  # Repeat for batch computation
    # Compute Wasserstein distances using Sinkhorn-Knopp
    wasserstein_distances = sinkhorn_knopp(pdfs_flat, true_density_flat,C, reg=0.05, n_iters=10)

    # Print results
    # print("Wasserstein distances:", wasserstein_distances)

    return torch.mean(wasserstein_distances)


def kl_divergence_r2(predicted_density, true_density, grid_x, grid_y, epsilon=1e-12):
    """
    Compute the KL divergence for discrete distributions defined on a 2D grid for a batch.
    Parameters:
    - predicted_density: torch.Tensor, shape (batch_size, num_grid_points_x, num_grid_points_y), predicted densities (P)
    - true_density: torch.Tensor, shape (batch_size, num_grid_points_x, num_grid_points_y), true densities (Q)
    - grid_x: torch.Tensor, shape (num_grid_points_x,), grid points corresponding to x-axis
    - grid_y: torch.Tensor, shape (num_grid_points_y,), grid points corresponding to y-axis
    - epsilon: float, small value to prevent numerical instability
    Returns:
    - kl_div: torch.Tensor, shape (batch_size,), KL divergence for each sample in the batch
    """
    # Add epsilon to avoid numerical issues
    P = predicted_density.clamp(min=epsilon)
    Q = true_density.clamp(min=epsilon)

    # Normalize P and Q to ensure valid probability distributions
    P = P / P.sum(dim=(1, 2), keepdim=True)
    Q = Q / Q.sum(dim=(1, 2), keepdim=True)

    # Compute the log term
    kl_terms = P * torch.log(P / Q)

    # Compute grid spacing
    grid_diff_x = torch.diff(grid_x, prepend=grid_x[..., :1])
    grid_diff_y = torch.diff(grid_y, prepend=grid_y[..., :1])

    # Compute KL divergence as the weighted sum of terms
    kl_div = torch.sum(kl_terms * grid_diff_x.unsqueeze(-1) * grid_diff_y.unsqueeze(0), dim=(1, 2))

    return torch.mean(kl_div)


def compute_weighted_mean(prob: torch.Tensor,
                          poses: torch.Tensor,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          theta: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted mean of a distribution
    :return mean of distribution
    """
    # Compute mean
    prob = prob.reshape(prob.shape[0], -1)
    prod = poses * prob[..., None]
    prod = prod.view(-1, x.size(0), x.size(1), x.size(2), 3)
    
    # Integrate x, y, theta
    int_x = torch.trapz(prod, x=x.unsqueeze(-1), dim=1)
    int_xy = torch.trapz(int_x, x=y[0, :, :].squeeze().unsqueeze(-1), dim=1)
    int_xyz = torch.trapz(int_xy, x=theta[0, 0, :].squeeze().unsqueeze(-1), dim=1)
    
    return int_xyz



def align(model, data):
    """
    Align two batched trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3, n, batch_size)
    data -- second trajectory (3, n, batch_size)

    Output:
    rot -- rotation matrix (3, 3, batch_size)
    trans -- translation vector (3, 1, batch_size)
    trans_error -- translational error per point (1, n, batch_size)
    """
    batch_size = model.shape[2]

    # Compute mean across points (dim=1) while keeping batch_size intact
    model_mean = model.mean(dim=1, keepdim=True)  # (3, 1, batch_size)
    data_mean = data.mean(dim=1, keepdim=True)    # (3, 1, batch_size)

    # Zero-center the data
    model_zerocentered = model - model_mean  # (3, n, batch_size)
    data_zerocentered = data - data_mean  # (3, n, batch_size)

    # Compute W efficiently (3, 3, batch_size)
    W = torch.einsum("jik,lik->jlk", model_zerocentered, data_zerocentered)  # (3, 3, batch_size)

    # Compute SVD for each batch
    U, d, Vh = torch.linalg.svd(W.permute(2, 0, 1))  # (batch_size, 3, 3)

    # Fix sign for proper rotation matrices
    S = torch.eye(3, dtype=model.dtype, device=model.device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 3, 3)
    dets = torch.det(U) * torch.det(Vh)  # (batch_size,)
    S[dets < 0, 2, 2] = -1  # Flip sign for batches where determinant is negative

    # Compute rotation for each batch
    rot = torch.bmm(U, torch.bmm(S, Vh)).permute(1, 2, 0)  # (3, 3, batch_size)

    # Compute translation for each batch
    trans = data_mean - torch.bmm(rot.permute(2, 0, 1), model_mean.permute(2, 0, 1)).permute(1, 2, 0) # (3, 1, batch_size)

    # Align model trajectory
    model_aligned = torch.bmm(rot.permute(2, 0, 1), model.permute(2, 0, 1)).permute(1, 2, 0) + trans  # (3, n, batch_size)

    # Compute translational error
    alignment_error = model_aligned - data  # (3, n, batch_size)
    trans_error = torch.sqrt(torch.sum(alignment_error ** 2, dim=0, keepdim=True))  # (1, n, batch_size)

    return rot, trans, trans_error



def rmse_se2(gt_trajectory_, trajectory_, scaling_factor=1.0, offset_x=0.0, offset_y=0.0):

    
    # Convert to torch tensors
    # gt_trajectory = gt_trajectory_[:, :, :2].T
    # trajectory = trajectory_[:, :, :2].T
    gt_trajectory = gt_trajectory_[:, :, :2].permute(2, 1, 0)
    trajectory = trajectory_[:, :, :2].permute(2, 1, 0)

    # Scale the gt trajectory
    gt_trajectory[0, :, :] = (gt_trajectory[0, :, :] / scaling_factor) - offset_x
    gt_trajectory[1, :, :] = (gt_trajectory[1, :, :] / scaling_factor) - offset_y
    zeros = torch.zeros((1, gt_trajectory.shape[1], gt_trajectory.shape[2]), dtype=torch.float32).to(gt_trajectory.device)

    # Append zeros in third dimension as z coordinate
    gt_trajectory = torch.vstack((gt_trajectory, zeros))

    # Scale second trajectory
    trajectory[0, :, :] = (trajectory[0, :, :] / scaling_factor) - offset_x
    trajectory[1, :, :] = (trajectory[1, :, :] / scaling_factor) - offset_y
    trajectory = torch.vstack((trajectory, zeros))

    # Align trajectory
    rot, trans, trans_error = align(trajectory, gt_trajectory)
    rot = rot.to(torch.float32)
    trans = trans.to(dtype=torch.float32)

    aligned_trajectory = torch.bmm(rot.permute(2, 0, 1), trajectory.permute(2, 0, 1)).permute(1, 2, 0) + trans  # (3, n, batch_size)

    # Compute metrics
    trans_error = trans_error.to(torch.float32)
    # metrics = torch.sqrt(torch.dot(trans_error, trans_error) / len(trans_error))

    # Assuming trans_error is of shape (1, 5, 10)
    trajectory_length = trans_error.shape[1]  # Infer the trajectory length dynamically

    trans_error_squared = trans_error ** 2  # Square the error (shape will be (1, 5, 10))

    # Sum along the trajectory dimension (axis=1) and then take the mean
    squared_error_sum = trans_error_squared.sum(dim=1)  # (1, 10) -- summed across trajectory length
    mean_squared_error = squared_error_sum / trajectory_length  # Normalize by trajectory length

    # Take the square root to get the final metric (1, 10) shape
    metrics = torch.sqrt(mean_squared_error).transpose(1,0)  # (1, 10) -- metric for each batch

    return torch.mean(metrics)

def kl_divergence( p, q):

    """
    Calculate the KL divergence between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return abs(torch.mean(torch.sum(p * torch.log(p / q), dim=(1, 2))))

def kl_divergence_s1( p, q):

    """
    Calculate the KL divergence between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return abs(torch.mean(torch.sum(p * torch.log(p / q),dim=-1)))

def error_s1(p,q):
    """
    Calculate the Absolute Error between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    error = abs(p-q)
    stacked_tensors = torch.stack((error, torch.ones_like(error)*2*math.pi - error), dim=-1)
    # Find the minimum values across the last dimension
    min_values, min_indices = torch.min(stacked_tensors, dim=-1)
    error_sign = p - q > 0
    signed_error = torch.where(
        min_indices == 0,  # If minimum index is 0
        min_values * (2 * error_sign.float() - 1),  # Apply original sign
        min_values * (1 - 2 * error_sign.float())  # Apply opposite sign
    )

    return signed_error

def absolute_error_s1(p,q):
    """
    Calculate the Absolute Error between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    if isinstance(p, torch.Tensor) or isinstance(q, torch.Tensor):
        error = torch.abs(p - q)
    else:
        error = abs(p - q)
    stacked_tensors = torch.stack((error, torch.ones_like(error)*2*math.pi - error), dim=-1)
    # Find the minimum values across the last dimension
    min_values, min_indices = torch.min(stacked_tensors, dim=-1)
    return min_values
def mean_absolute_error(p, q):
    """
    Calculate the Mean Absolute Error (MAE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    min_values = absolute_error_s1(p, q)
    return torch.mean(min_values)

def root_mean_square_error(p, q):
    """
    Calculate the Root Mean Square Error (RMSE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    return torch.sqrt(torch.mean((p - q) ** 2,))


def root_mean_square_error_s1(p, q):
    """
    Calculate the Root Mean Square Error (RMSE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    min_values = absolute_error_s1(p, q)
    return torch.mean(torch.sqrt(torch.mean(min_values ** 2,dim=(-1))))

def wasserstein_distance(p, q):
    """
    Calculate the Wasserstein distance (Earth Mover's Distance) between two 1D distributions p and q in a batch.
    Both p and q should be 2D torch tensors of the same shape, where the first dimension is the batch size.
    """
    p_sorted, _ = torch.sort(p, dim=-1)
    q_sorted, _ = torch.sort(q, dim=-1)
    return torch.sum(torch.abs(p_sorted - q_sorted), dim=-1).mean()

def kl_divergence_k(predicted_density, true_density, grid, epsilon=1e-12):
    """ 
    Compute the KL divergence for discrete distributions defined on a grid for a batch.
    Parameters:
    - predicted_density: torch.Tensor, shape (batch_size, num_grid_points), predicted densities (P)
    - true_density: torch.Tensor, shape (batch_size, num_grid_points), true densities (Q)
    - grid: torch.Tensor, shape (num_grid_points,), grid points corresponding to densities
    - epsilon: float, small value to prevent numerical instability
    Returns:
    - kl_div: torch.Tensor, shape (batch_size,), KL divergence for each sample in the batch
    """
    # Add epsilon to avoid numerical issues
    P = predicted_density.clamp(min=epsilon)
    Q = true_density.clamp(min=epsilon)

    # Normalize P and Q to ensure valid probability distributions
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)

    # Compute the log term and handle grid spacing
    kl_terms = P * torch.log(P / Q)
    grid_diff = torch.diff(grid, prepend=grid[..., :1])  # Compute spacing between grid points

    # Compute KL divergence as the weighted sum of terms
    kl_div = torch.sum(kl_terms * grid_diff, dim=-1)

    return torch.mean(kl_div)


def predicted_residual_error(p, q, bins=10):
    """
    Calculate the predicted residual error between two tensors p and q and store the frequency of the values in bins.
    Both p and q should be torch tensors of the same shape.
    """
    residual_error = absolute_error_s1(p, q)
    if torch.isnan(residual_error).any():
        print("residual error is nan")
    hist = torch.histc(residual_error, bins=bins)
    min_val, max_val = torch.min(residual_error), torch.max(residual_error)
    bin_edges = torch.linspace(min_val, max_val, 20 + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return residual_error, hist, bin_centers, bin_edges


def expected_calibration_error_continuous(y_true, y_pred_cdf, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for continuous distributions.
    
    Args:
        y_true (torch.Tensor): Values of the true distribution pdf
        y_pred_cdf (torch.Tensor): Predicted CDF values corresponding to y_true, shape (n_samples,).
        n_bins (int): Number of bins for the calibration evaluation.
    
    Returns:
        float: The ECE score.
    """
    # Define bin edges based on observed data
    min_val, max_val = torch.min(y_true), torch.max(y_true)
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1)

    # Initialize ECE and bin population
    ece = 0.0
    total_samples = y_true.size(1)
    
    for i in range(n_bins):
        # Get the bin range
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        # pdb.set_trace()
        # Find samples in the current bin
        in_bin = (y_true >= lower_edge) & (y_true < upper_edge)
        n_in_bin = torch.sum(in_bin,dim=1)
        
        # if n_in_bin == 0:
        #     continue  # Skip empty bins
        
        # Compute empirical CDF for the bin
        empirical_cdf = torch.mean((y_true <= upper_edge).float(), dim=1)

        # Compute predicted CDF for the bin
        predicted_cdf = torch.mean(torch.where(in_bin, y_pred_cdf, torch.tensor(0)), dim=1)
        # Compute the bin's contribution to ECE
        ece += n_in_bin / total_samples * torch.abs(predicted_cdf - empirical_cdf)
    
    return torch.mean(ece)

# # Example usage
# # Observed data (true values)
# y_true = torch.tensor([0.1, 0.4, 0.5, 0.8, 0.9])

# # Predicted CDF values corresponding to y_true
# # These could come from a probabilistic model, e.g., a normal distribution
# y_pred_cdf = torch.tensor([0.2, 0.5, 0.6, 0.7, 0.95])

# # Compute ECE
# ece_score = expected_calibration_error_continuous(y_true, y_pred_cdf, n_bins=5)
# print(f"Expected Calibration Error (ECE): {ece_score}")

def expected_calibration_error(predicted_distribution, true_distribution, M=5):
    """
    Computes the Expected Calibration Error (ECE) between the predicted and true distributions.
    ECE is a scalar measure of how well the predicted probabilities are calibrated with respect to the true outcomes.
    It is calculated by partitioning the predictions into M bins and computing the weighted average of the absolute 
    difference between the accuracy and confidence of each bin.
    Args:
        predicted_distribution (torch.Tensor): A tensor of predicted probabilities with shape (batch_size, num_samples).
        true_distribution (torch.Tensor): A tensor of true probabilities with shape (batch_size, num_samples).
        M (int, optional): The number of bins to partition the predicted probabilities into. Default is 5.
    Returns:
        torch.Tensor: A tensor containing the ECE for each batch element.
    """
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    total_samples = predicted_distribution.shape[-1]
    batch_size = predicted_distribution.shape[0]
    ece = torch.zeros(batch_size)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        in_bin_pred = torch.logical_and(predicted_distribution > bin_lower, predicted_distribution <= bin_upper)
        # Identify samples where the true PDF is also within the bin
        in_bin_true = torch.logical_and(true_distribution > bin_lower, true_distribution <= bin_upper)
        
        # Determine the samples that satisfy both conditions
        matched_samples = torch.logical_and(in_bin_pred, in_bin_true)
        n_in_bin = torch.sum(in_bin_pred,dim=1)
        if torch.any(n_in_bin > 0):
            non_zero_bins = n_in_bin > 0
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = torch.where(non_zero_bins, matched_samples.sum(dim=-1) / n_in_bin, torch.tensor(0.0))
            # get the average confidence of bin m: conf(Bm)
            masked_tensor = torch.where(in_bin_pred, predicted_distribution, torch.tensor(0.0))
            avg_confidence_in_bin = torch.where(non_zero_bins, masked_tensor.sum(dim=-1) / n_in_bin, torch.tensor(0.0))
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.where(non_zero_bins, abs(avg_confidence_in_bin - accuracy_in_bin) * (n_in_bin / total_samples), torch.tensor(0.0))

    return torch.mean(ece)

def expected_calibration_error_check(predicted_distribution, true_distribution, M=5):
    """
    Computes the Expected Calibration Error (ECE) between the predicted and true distributions.
    """
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    batch_size, num_samples = predicted_distribution.shape
    ece = torch.zeros(batch_size, device=predicted_distribution.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Mask for samples in the current bin
        in_bin_pred = (predicted_distribution > bin_lower) & (predicted_distribution <= bin_upper)
        in_bin_true = (true_distribution > bin_lower) & (true_distribution <= bin_upper)
        matched_samples = in_bin_pred & in_bin_true
        
        # Count samples in the bin
        n_in_bin = in_bin_pred.sum(dim=-1)
        
        # Avoid division by zero
        non_zero_bins = n_in_bin > 0
        
        # Calculate accuracy and confidence
        accuracy_in_bin = torch.zeros_like(ece)
        avg_confidence_in_bin = torch.zeros_like(ece)
        
        if non_zero_bins.any():
            accuracy_in_bin[non_zero_bins] = matched_samples.sum(dim=-1)[non_zero_bins] / n_in_bin[non_zero_bins]
            masked_tensor = torch.where(in_bin_pred, predicted_distribution, torch.tensor(0.0))
            avg_confidence_in_bin[non_zero_bins] = masked_tensor.view(batch_size, -1).mean(dim=-1)[non_zero_bins]
        
        # Weighted absolute difference
        ece += torch.where(
            non_zero_bins,
            torch.abs(avg_confidence_in_bin - accuracy_in_bin) * (n_in_bin / num_samples),
            torch.tensor(0.0, device=ece.device)
        )
    
    return ece.mean()
def sharpness_discrete(pred_probs):
    """
    Compute sharpness for discrete distributions based on entropy.
    
    Parameters:
    - pred_probs: torch.Tensor, predicted probabilities (N, K), where K is the number of classes.
    
    Returns:
    - sharpness: float, the average sharpness metric.
    """
    entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-9), dim=1)  # Add epsilon to avoid log(0)
    return torch.mean(entropy)

def compute_cdf_from_pdf(pdf):
    """
    Compute the CDF from the given PDF values.

    Args:
        pdf (torch.Tensor): PDF values over the support, shape (n_samples,).

    Returns:
        torch.Tensor: CDF values over the same support, shape (n_samples,).
    """
    # Ensure the PDF is normalized
    pdf_sum = torch.sum(pdf)
    if not torch.isclose(pdf_sum, torch.tensor(1.0), atol=1e-6):
        pdf = pdf / pdf_sum

    # Compute the cumulative sum to get the CDF
    cdf = torch.cumsum(pdf, dim=0)
    return cdf

# # Example PDF values
# pdf = torch.tensor([0.1, 0.2, 0.3, 0.4])

# # Compute the CDF
# cdf = compute_cdf_from_pdf(pdf)
# print("PDF:", pdf)
# print("CDF:", cdf)



