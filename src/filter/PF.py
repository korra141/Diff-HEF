"""
A class that implements a batched particle filter using PyTorch.
"""
from typing import Tuple

import torch
import pdb
from torch.distributions import Normal
# from src.utils.door_dataset_utils import preprocess_mask

from einops import rearrange, repeat


class RangePF:
    def __init__(self,
                 prior_mu: torch.Tensor,
                 prior_cov: torch.Tensor,
                 n_particles: int = 100,
                 batch_size: int = 1,
                 grid_size = (100, 100, 36),
                 grid_bounds = (-0.5, 0.5),
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        :param prior: a prior pose of as a torch tensor of dimension (B, 3) where B is the batch size
        :param prior_cov: Covariance noise for the prior distribution of dimension (B, 3, 3) or (3, 3)
        :param n_particles: Number of particles to use.
        :param batch_size: Number of independent particle filters to run in parallel
        :param device: Device to use for tensor operations.
        """
        self._N = n_particles
        self.batch_size = batch_size
        self.device = device
        
        # Move tensors to specified device
        prior_mu = prior_mu.to(device)
        prior_cov = prior_cov.to(device)
        
        # Ensure proper batch dimension
        if prior_mu.ndim == 1:
            prior_mu = prior_mu.unsqueeze(0).expand(batch_size, -1)
        
        if prior_cov.ndim == 2:
            prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate particles from prior: (B, N, 3)
        L = torch.linalg.cholesky(prior_cov)  # (B, 3, 3)
        noise = torch.randn((batch_size, 3, n_particles), device=device, dtype=torch.float64)  # (B, 3, N)
        
        # L @ noise -> (B, 3, N), then transpose and add prior
        particles_noise = torch.bmm(L, noise)  # (B, 3, N)
        self.particles = particles_noise.permute(0, 2, 1) + prior_mu.unsqueeze(1)  # (B, N, 3)
        
        # Initialize weights: (B, N)
        self.weights = torch.ones((batch_size, n_particles), device=device) / n_particles
        self.mode_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds

    def prediction(self,
                   step: torch.Tensor,
                   step_cov: torch.Tensor) -> None:
        """
        Prediction step PF
        :param step: motion step (relative displacement) of dimension (B, 3)
        :param step_cov: Covariance matrix of prediction step of dimension (B, 3, 3) or (3, 3)
        """
        # Ensure tensors are on correct device
        step = step.to(self.device)
        step_cov = step_cov.to(self.device)
        
        # Ensure proper batch dimension
        if step.ndim == 1:
            step = step.unsqueeze(0).expand(self.batch_size, -1)
        
        if step_cov.ndim == 2:
            step_cov = step_cov.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Sample steps: (B, N, 3)
        L = torch.linalg.cholesky(step_cov)  # (B, 3, 3)
        noise = torch.randn((self.batch_size, 3, self._N), device=self.device, dtype=torch.float64)  # (B, 3, N)
        step_sample = torch.bmm(L, noise).permute(0, 2, 1) + step.unsqueeze(1)  # (B, N, 3)
        
        # Normalize angles
        step_sample[:, :, 2] = (step_sample[:, :, 2] + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Apply step
        c = torch.cos(self.particles[:, :, 2])  # (B, N)
        s = torch.sin(self.particles[:, :, 2])  # (B, N)
        
        # Update positions
        self.particles[:, :, 0] += c * step_sample[:, :, 0] - s * step_sample[:, :, 1]
        self.particles[:, :, 1] += s * step_sample[:, :, 0] + c * step_sample[:, :, 1]
        self.particles[:, :, 2] += step_sample[:, :, 2]
        
        # Normalize angles
        self.particles[:, :, 2] = (self.particles[:, :, 2] + torch.pi) % (2 * torch.pi) - torch.pi

        return self.particles

    def update(self, log_weights) -> torch.Tensor:
        """
        Update step PF
        :param landmarks: location of each UWB landmark in the map (L, 2) or (B, L, 2)
        :param observations: range measurements of dimension (B, L)
        :param observations_cov: variance of each measurement of dimension (B, L) or (L,)
        :return Mean of the particles (B, 3)
        """
        
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        self.weights = torch.exp(log_weights)
        # self.weights = weights

         # Add weight validation
        if torch.isnan(self.weights).any():
            print(f"Warning: Found nan in weights after normalization")
            self.weights = torch.nan_to_num(self.weights, nan=1.0/self._N)
            self.weights /= self.weights.sum(dim=1, keepdim=True)
        
        # Resample for each batch
        # new_particles = []
        for b in range(self.batch_size):
            # Compute cumulative sum: (N,)
            cumulative_sum = torch.cumsum(self.weights[b], dim=0)
            cumulative_sum[-1] = 1.0  # Avoid round-off error

            cumulative_sum = cumulative_sum.to(self.device)
            # print(cumulative_sum.min(), cumulative_sum.max())
            # Low variance sampling
            r = torch.rand(1, device=self.device) / self._N
            samples = torch.linspace(0.0, 1.0, steps=self._N, device=self.device) + r
            # print(samples.min(), samples.max())
            # samples = torch.clamp(samples, min=0.0, max=1.0)
            # Find indices for resampling: (N,)
            indexes = torch.searchsorted(cumulative_sum, samples)

            indexes = torch.clamp(indexes, min=0, max=self._N - 1)
            # new_particles.append(self.particles[b, indexes].clone())
            # Resample particles for this batch
            self.particles[b] = self.particles[b, indexes]

            # Store mode index
            self.mode_index[b] = torch.argmax(self.weights[b]).item()
            # Reset weights
            self.weights[b].fill_(1.0 / self._N)
        # self.particles = torch.stack(new_particles)
        # Return mean of particles for each batch: (B, 3)
        return torch.mean(self.particles, dim=1)

    def compute_mode(self) -> torch.Tensor:
        """
        Compute mode of the distribution
        :return mode of distribution (B, 3)
        """
        # Collect mode particles using mode_index for each batch
        batch_indices = torch.arange(self.batch_size, device=self.device)
        return self.particles[batch_indices, self.mode_index]


    def histogramdd_pytorch(self,particles, x_bins, y_bins, theta_bins, device):
        # Ensure the bins are on the same device
        x_bins = x_bins.to(device)
        y_bins = y_bins.to(device)
        theta_bins = theta_bins.to(device)

        # Compute the bin edges
        # For each dimension, extend the edges by half the bin width on both sides
        x_edges = torch.cat([x_bins[0:1] - (x_bins[1] - x_bins[0]) / 2, x_bins + (x_bins[1] - x_bins[0]) / 2])
        y_edges = torch.cat([y_bins[0:1] - (y_bins[1] - y_bins[0]) / 2, y_bins + (y_bins[1] - y_bins[0]) / 2])
        theta_edges = torch.cat([theta_bins[0:1] - (theta_bins[1] - theta_bins[0]) / 2, theta_bins + (theta_bins[1] - theta_bins[0]) / 2])

        # Compute histograms for each dimension (x, y, theta)
        hist_x = torch.histc(particles[:, 0], bins=x_bins.size(0)-1, min=x_edges[0], max=x_edges[-1])
        hist_y = torch.histc(particles[:, 1], bins=y_bins.size(0) -1, min=y_edges[0], max=y_edges[-1])
        hist_theta = torch.histc(particles[:, 2], bins=theta_bins.size(0) -1, min=theta_edges[0], max=theta_edges[-1])
        
        # Normalize histograms by dividing by the total count to get densities
        hist_x = hist_x / hist_x.sum()
        hist_y = hist_y / hist_y.sum()
        hist_theta = hist_theta / hist_theta.sum()

        # Reshape histograms to 3D grid
        hist_x = hist_x.view(-1, 1, 1)
        hist_y = hist_y.view(1, -1, 1)
        hist_theta = hist_theta.view(1, 1, -1)

        # Multiply histograms to compute the joint density
        density = hist_x * hist_y * hist_theta

        return density


    def neg_log_likelihood(self, 
                          pose: torch.Tensor) -> torch.Tensor:
        """
        Evaluate posterior distribution of histogram filter
        :param pose: Pose at which to interpolate the SE2 Fourier transform (B, 3)
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        :return ll: Negative log likelihood at given pose (B,)
        """
        pose = pose.to(self.device)

        pose[:, 2] = (pose[:, 2] + 2*torch.pi) % (2 * torch.pi)


        
        # Define bins
        x_bins = torch.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[0] + 1, device=self.device)
        y_bins = torch.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[1] + 1, device=self.device)
        theta_bins = torch.linspace(-torch.pi, torch.pi, self.grid_size[2] + 1, device=self.device)
        # theta_bins = torch.linspace(0, 2*torch.pi, self.grid_size[2] + 1, device=self.device)
        
        # Initialize result tensor
        neg_ll = torch.zeros(self.batch_size, device=self.device)
        
        # Process each batch separately since histogramming isn't easily batchable
        for b in range(self.batch_size):
            # Convert particles for this batch to CPU for histogram computation
            particles_cpu = self.particles[b].cpu().numpy()
            # particles_cpu[:,2] = (particles_cpu[:,2] + 2*torch.pi) % (2 * torch.pi)
            # pdb.set_trace()
            # Compute density using numpy
            import numpy as np
            density, _ = np.histogramdd(particles_cpu,
                                        bins=(x_bins.cpu().numpy().flatten(), y_bins.cpu().numpy().flatten(), theta_bins.cpu().numpy().flatten()),
                                        density=True)
            density = torch.tensor(density, device=self.device)

            # density = self.histogramdd_pytorch(self.particles[b], x_bins, y_bins, theta_bins, self.device)
            
            # Find the closest bin to current pose
            x_index = torch.searchsorted(x_bins[:-1], pose[b, 0], right=True) - 1
            y_index = torch.searchsorted(y_bins[:-1], pose[b, 1], right=True) - 1
            theta_index = torch.searchsorted(theta_bins[:-1], pose[b, 2], right=True) - 1
            
    
            p_g = torch.where(torch.isnan(density[x_index, y_index, theta_index]), torch.tensor(0.0, device=self.device), density[x_index, y_index, theta_index])
            
            # Prevent zero likelihood error
            ll = torch.log(p_g + 1e-8)
            neg_ll[b] = -ll
            
        return torch.mean(neg_ll)

    def weight_density(self, weights, grid, device):
        """
        Compute the density from weights on a 1D grid.

        Args:
            weights (torch.Tensor): 1D tensor of weights, shape [N].
            grid (torch.Tensor): 1D tensor representing grid points, shape [M].
            device (torch.device): Device to perform computations on.

        Returns:
            torch.Tensor: Density estimated over the grid.
        """
        # Ensure grid and weights are on the same device
        grid = grid.to(device)
        weights = weights.to(device)

        # Compute the grid edges (min, max)
        grid_edges = torch.cat([grid[0:1] - (grid[1] - grid[0]) / 2, grid + (grid[1] - grid[0]) / 2])

        # Compute the histogram-like density from weights using grid
        density = torch.zeros(grid.size(0) - 1, device=device)

        # Loop through weights and sum over the grid intervals
        for i in range(1, grid.size(0)):
            # Find indices of weights within this grid segment
            segment_mask = (weights >= grid_edges[i-1]) & (weights < grid_edges[i])
            density[i-1] = segment_mask.sum().float() / (grid[1] - grid[0])  # Normalize by grid step

        # Normalize density to sum to 1 (optional)
        density = density / density.sum()

        return density


    def neg_log_likelihood_measurement(self, poses:torch.Tensor, range_beacon:torch.Tensor, measurement: torch.Tensor, measurement_pdf:torch.Tensor) -> torch.Tensor:
        """
        Evaluate posterior distribution of histogram filter
        :param pose: Pose at which to interpolate the SE2 Fourier transform (B, 3)
        :param grid_bounds: x-y bounds of the grid, this assumes a square grid
        :param grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
        :return ll: Negative log likelihood at given pose (B,)
        """

        # grid is defined 
        grid = torch.linalg.norm(range_beacon  - poses[:, :, 0:2], dim=-1) #[batch_size, N] 

        # Initialize result tensor
        neg_ll = torch.zeros(self.batch_size, device=self.device)
        
        # Process each batch separately since histogramming isn't easily batchable
        for b in range(self.batch_size):
    
            density = self.weight_density(measurement_pdf[b], grid[b], self.device)
            
            # Get the measurement for the current batch element (shape [1])
            current_measurement = measurement[b]  # shape [1]
            
            # Get the corresponding grid for this batch element (shape [N])
            current_grid = grid[b]  # shape [N]
            
            # Find the closest grid value to the current measurement
            closest_index = torch.searchsorted(current_grid, current_measurement, right=True) - 1
            
            # Ensure that the index is within bounds
            if 0 <= closest_index < current_grid.size(0):
                # Get the density corresponding to the closest grid point (e.g., you would have a density tensor)
                p_g = density[closest_index]  # Assuming density is a 1D tensor corresponding to grid values
            else:
                # If the index is out of bounds, set p_g to 0 or some default value
                p_g = torch.tensor(0.0, device=self.device)
            # Prevent zero likelihood error
            ll = torch.log(p_g + 1e-8)
            neg_ll[b] = -ll
            
        return torch.mean(neg_ll)


class BatchedBearingPF(RangePF):
    def __init__(self, d_door2pose: float = 0.1, **kwargs):
        """
        :param d_door2pose: Distance from door to pose
        :param kwargs: Arguments passed to BatchedRangePF
        """
        super().__init__(**kwargs)
        self.d_door2pose = d_door2pose

    def update(self,
               landmarks: torch.Tensor,
               map_mask: torch.Tensor,
               observations: torch.Tensor,
               observations_cov: torch.Tensor) -> torch.Tensor:
        """
        Update step PF for bearing-only measurements
        :param landmarks: location of all doors in the map (x-y coordinates) (L, 2) or (B, L, 2)
        :param map_mask: Binary mask indicating traversable area (B, H, W) or (H, W)
        :param observations: bearing measurements of dimension (B, M)
        :param observations_cov: variance of each door of dimension (B, L) or (L,)
        :return Mean of the particles (B, 3)
        """
        # Ensure tensors are on correct device
        landmarks = landmarks.to(self.device)
        if isinstance(map_mask, torch.Tensor):
            map_mask = map_mask.to(self.device)
        observations = observations.to(self.device)
        observations_cov = observations_cov.to(self.device)
        
        # Ensure proper batch dimension for landmarks
        if landmarks.ndim == 2:
            landmarks = landmarks.unsqueeze(0).expand(self.batch_size, -1, -1)  # (B, L, 2)
        
        # Ensure proper batch dimension for observations_cov
        if observations_cov.ndim == 1:
            observations_cov = observations_cov.unsqueeze(0).expand(self.batch_size, -1)  # (B, L)
        
        observations_std = torch.sqrt(observations_cov)  # (B, L)
        
        # Map weights temporarily to log space: (B, N)
        log_weights = torch.log(self.weights + 1e-9)
        
        # Initialize masks for each batch
        if isinstance(map_mask, torch.Tensor):
            # Handle single mask or batch of masks
            if map_mask.ndim == 2:  # Single mask
                # Process mask for each batch separately
                masks = []
                for b in range(self.batch_size):
                    batch_mask = preprocess_mask(map_mask, self.particles[b])
                    if not isinstance(batch_mask, torch.Tensor):
                        batch_mask = torch.tensor(batch_mask, device=self.device)
                    masks.append(batch_mask)
                mask = torch.stack(masks)  # (B, N)
            else:  # Batch of masks
                # Process each mask with corresponding particles
                masks = []
                for b in range(self.batch_size):
                    batch_mask = preprocess_mask(map_mask[b], self.particles[b])
                    if not isinstance(batch_mask, torch.Tensor):
                        batch_mask = torch.tensor(batch_mask, device=self.device)
                    masks.append(batch_mask)
                mask = torch.stack(masks)  # (B, N)
        else:
            mask = torch.ones((self.batch_size, self._N), device=self.device)
        
        # Process each measurement
        num_observations = observations.shape[1]
        
        for i in range(num_observations):
            # Current observation for all batches: (B,)
            obs_i = observations[:, i]
            
            # For each batch and landmark, compute bearing angle
            # Reshape landmarks: (B, L, 2, 1) and particles: (B, 1, N, 2)
            landmarks_expanded = landmarks.unsqueeze(3)  # (B, L, 2, 1)
            particles_xy = self.particles[:, :, :2].unsqueeze(1)  # (B, 1, N, 2)
            
            # Compute difference: (B, L, N, 2)
            diff = landmarks_expanded - particles_xy
            
            # Compute angle: (B, L, N)
            angle = torch.atan2(diff[:, :, :, 1], diff[:, :, :, 0])
            
            # Adjust by particle orientation: (B, L, N)
            particles_theta = self.particles[:, :, 2].unsqueeze(1)  # (B, 1, N)
            angle = angle - particles_theta
            
            # Wrap angle: (B, L, N)
            angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
            
            # Compute angle difference: (B, L, N)
            obs_expanded = obs_i.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            diff_angle = obs_expanded - angle
            diff_angle = (diff_angle + torch.pi) % (2 * torch.pi) - torch.pi
            
            # Compute likelihood for each landmark-particle pair
            # std_expanded: (B, L, 1)
            std_expanded = observations_std[:, :, None]
            
            # Normal distribution: (B, L, N)
            normal_dist = Normal(0.0, std_expanded)
            mixture = mask.unsqueeze(1) * normal_dist.log_prob(diff_angle).exp()
            
            # Max along landmarks dimension: (B, N)
            mixture_max = mixture.max(dim=1)[0] + 1e-8
            
            # Update log weights: (B, N)
            log_weights = torch.maximum(log_weights, torch.log(mixture_max))
        
        # Mask out particles outside map: (B, N)
        log_weights = torch.where(mask == 0, torch.tensor(-float('inf'), device=self.device), log_weights)
        
        # Normalize weights batch-wise: (B, N)
        log_weights -= torch.logsumexp(log_weights, dim=1, keepdim=True)
        self.weights = torch.exp(log_weights)
        
        # Resample for each batch
        for b in range(self.batch_size):
            # Compute cumulative sum: (N,)
            cumulative_sum = torch.cumsum(self.weights[b], dim=0)
            cumulative_sum[-1] = 1.0  # Avoid round-off error

            
            # Low variance sampling
            r = torch.rand(1, device=self.device) / self._N
            # r = torch.rand(1) / self._N
            samples = torch.linspace(0.0, 1.0, steps=self._N, device=self.device) + r
            # samples = torch.linspace(0.0, 1.0, steps=self._N) + r
            
            # Find indices for resampling: (N,)
            indexes = torch.searchsorted(cumulative_sum, samples)
            
            # Resample particles for this batch
            self.particles[b] = self.particles[b, indexes].clone()
            
            # Store mode index
            self.mode_index[b] = torch.argmax(self.weights[b]).item()
            
            # Reset weights
            self.weights[b].fill_(1.0 / self._N)
        
        # Return mean of particles for each batch: (B, 3)
        return torch.mean(self.particles, dim=1)