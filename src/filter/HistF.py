"""
A PyTorch implementation of a batched range-only histogram filter
"""
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
# from torchvision.transforms.functional import gaussian_blur
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import pdb
from src.utils.metrics import compute_weighted_mean


class BatchedRangeHF:
    def __init__(self,
                 batch_size: int,
                 prior_mu: torch.Tensor,
                 prior_cov: torch.Tensor,
                 grid_samples: torch.Tensor,
                 x : torch.Tensor,
                 y : torch.Tensor,
                 theta : torch.Tensor,
                 grid_bounds: Tuple[float] = (-0.5, 0.5),
                 grid_size: Tuple[int] = (50, 50, 32),
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Batched Histogram Filter for Range-based localization.
        
        Args:
            batch_size: Number of parallel filters to maintain
            prior: Prior pose as a tensor of dimension (batch_size, 3)
            prior_cov: Covariance noise for the prior distribution (batch_size, 3, 3) or (3, 3)
            grid_samples: Samples of the grid (x, y, theta) of shape (N, 3) where N = x_dim * y_dim * theta_dim
            grid_bounds: x-y bounds of the grid, assumes a square grid
            grid_size: Dimensions of the grid (x_dim, y_dim, theta_dim)
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.device = device
        self.batch_size = batch_size
        
        # Move tensors to device
        self.grid_samples = grid_samples.to(device)
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size
        self._N = torch.prod(torch.tensor(grid_size))
        self.x = x
        self.y = y
        self.theta = theta   
        # Calculate grid steps
        self.step_xy = (grid_bounds[1] - grid_bounds[0]) / grid_size[0]
        self.step_theta = (torch.pi * 2) / grid_size[2]
        self.volume = self.step_xy * self.step_xy * self.step_theta
        
        # Ensure prior_cov is batched
        if prior_cov.dim() == 2:
            prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Define prior and normalize it (batched)
        self.prior = self._compute_multivariate_normal_pdf(grid_samples, prior_mu, prior_cov)
        self.prior = self.prior / torch.sum(self.prior, dim=1, keepdim=True)
    
    def _compute_multivariate_normal_pdf(self, 
                                         x: torch.Tensor, 
                                         mean: torch.Tensor, 
                                         cov: torch.Tensor) -> torch.Tensor:
        """
        Compute multivariate normal PDF for batched inputs.
        
        Args:
            x: Points to evaluate, shape (N, D)
            mean: Mean of distribution, shape (batch_size, D)
            cov: Covariance matrices, shape (batch_size, D, D)
            
        Returns:
            PDF values at x for each batch, shape (batch_size, N)
        """
        batch_size, dim = mean.shape
        N = x.shape[0]
        
        # Expand x to match batch dimension: (N, D) -> (batch_size, N, D)
        # x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Expand mean: (batch_size, D) -> (batch_size, 1, D)
        mean_expanded = mean.unsqueeze(1)
        
        # Calculate x - mean for each batch: (batch_size, N, D)
        diff = x - mean_expanded

        diff[:, :,  2] = (diff[:, :,  2] + torch.pi) % (2 * torch.pi) - torch.pi
        
        # For each batch, compute (x - mean)^T * cov^-1 * (x - mean)
        # First compute cov inverse for each batch
        cov_inv = torch.inverse(cov).unsqueeze(1)  # (batch_size, 1,  D, D)
        
        # Reshape for batch matrix multiplication
        # cov_inv = cov_inv.unsqueeze(1).expand(batch_size, N, dim, dim)
        diff_reshaped = diff.unsqueeze(-1)  # (batch_size, N, D, 1)
        
        # Matrix multiplication
        mahalanobis_dist = torch.matmul(
            torch.matmul(diff_reshaped.transpose(-2, -1), cov_inv),
            diff_reshaped
        ).squeeze(-1).squeeze(-1)  # (batch_size, N)
        
        # Compute normalization factor
        det = torch.det(cov)
        norm_factor = 1.0 / (torch.sqrt((2 * torch.pi) ** dim * det)).unsqueeze(1)
        
        # Compute PDF
        pdf = norm_factor * torch.exp(-0.5 * mahalanobis_dist)
        
        return pdf
    
    def prediction(self,
                   step: torch.Tensor,
                   step_cov: torch.Tensor) -> None:
        """
        Prediction step for batched HF.
        
        Args:
            step: Motion step (relative displacement) of dimension (batch_size, 3)
            step_cov: Covariance matrix of prediction step (batch_size, 3, 3) or (3, 3)
        """
        # Ensure step_cov is batched
        if step_cov.dim() == 2:
            step_cov = step_cov.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Apply step to obtain expected transition for each batch
        # centroids = self.grid_samples.clone().unsqueeze(0).expand(self.batch_size, -1, -1)  # (batch_size, N, 3)
        centroids = self.grid_samples.clone()
        
        c = torch.cos(centroids[..., 2])
        s = torch.sin(centroids[..., 2])
        
        # Expand step for broadcasting
        step_expanded = step.unsqueeze(1)  # (batch_size, 1, 3)
        
        # Update centroids with motion
        centroids[..., 0] += c * step_expanded[..., 0] - s * step_expanded[..., 1]
        centroids[..., 1] += s * step_expanded[..., 0] + c * step_expanded[..., 1]
        centroids[..., 2] += step_expanded[..., 2]
        centroids[..., 2] = (centroids[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        
        # for i in range(self.batch_size):
        bins = self._compute_bin_indices(centroids)
        mask = self._filter_out_of_bounds(centroids)
    
        # Update prior belief for each batch
        reshaped_prior = self.prior.reshape(self.batch_size, *self.grid_size)
        belief = torch.zeros_like(reshaped_prior)

        # Compute linear indices for bins
        grid_size = self.grid_size  # Assuming grid_size is [H, W, D]
        # linear_bins = bins[:, :, 0] * (grid_size[1] * grid_size[2]) + bins[:, :, 1] * grid_size[2] + bins[:, :, 2]

        # # Flatten belief and mask
        # belief_flat = belief.view(self.batch_size, -1)
        # prior_flat = (self.prior * mask).view(self.batch_size, -1)

        # # Add values to belief using index_add_
        # for batch_idx in range(self.batch_size):
        #     belief_flat[batch_idx].index_add_(0, linear_bins[batch_idx].flatten(), prior_flat[batch_idx].flatten())

        # # Reshape belief back to original shape
        # belief = belief_flat.view(self.batch_size, *self.grid_size)

        scaled_sigma = torch.sqrt(step_cov)/ torch.tensor([self.step_xy, self.step_xy, self.step_theta]).to(step_cov.device)
        # # Ensure scaled_sigma matches the rank of belief
        scaled_sigma_np = scaled_sigma.cpu().numpy()  # Convert to numpy array
        prior_np = self.prior.cpu().numpy()
        belief_np_filter = np.zeros_like(belief.cpu().numpy())
        belief_np = belief.cpu().numpy()
        for i in range(self.batch_size):
            np.add.at(belief_np[i], tuple(bins[i].T.cpu().numpy()), prior_np[i] * mask[i].cpu().numpy())
            belief_np_filter[i] = gaussian_filter(belief_np[i], sigma=np.diag(scaled_sigma_np[i]), mode="constant")

        # Convert back to PyTorch tensor
        belief = torch.tensor(belief_np_filter, device=step.device)
        
        # Normalize belief for each batch
        belief = belief / torch.sum(belief.reshape(self.batch_size, -1), dim=1).reshape(self.batch_size, 1, 1, 1)
        
        # Update prior if there is probability mass
        self.prior = belief.reshape(self.batch_size, -1)

        return belief.reshape(self.batch_size, -1)
    
    def update(self,energy) -> torch.Tensor:
        """
        Update step for batched HF.
        
        Args:
            landmarks: Location of each UWB landmark (n_landmarks, 3)
            observations: Range measurements (batch_size, n_landmarks)
            observations_cov: Variance of each measurement (batch_size, 3, 3) or (n_landmarks,)
            
        Returns:
            Mean pose for each batch (batch_size, 3)
        """
        # print(f"Histogram Filter update prior shape: {self.prior.shape}") #[batch_size, N]
        # print(f"Histogram Filter update landmarks shape: {landmarks.shape}") #[batch_size, n_landmarks, 2]
        # print(f"Histogram Filter update observations shape: {observations.shape}") #[batch_size,1]
        # print(f"Histogram Filter update observations_std shape: {observations_std.shape}") #[batch_size, 1,1] or [n_landmarks]
        # observations_std = torch.sqrt(observations_cov)
        
        # dist = torch.linalg.norm(landmarks  - self.grid_samples[:, :, 0:2], dim=-1)
        # energy = torch.distributions.Normal(observations, observations_std).log_prob(dist)
        # # energy = energy.reshape(-1, *args.grid_size)
        # # print(f"Histogram Filter update energy shape: {energy.shape}") #[batch_size, N]
                
        # # Normalize log likelihoods
        # energy -= torch.logsumexp(energy, dim=0)

        posterior = self.prior * torch.exp(energy)
        # posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True))
        if torch.sum(posterior) != 0:
            posterior /= torch.sum(posterior, dim=-1, keepdim=True)
        
        # Update prior if there is probability mass
        self.prior = posterior
        posterior = posterior.view(self.batch_size, *self.grid_size)

        means = self._compute_mean()
        # means = compute_weighted_mean(posterior, self.grid_samples, self.x, self.y, self.theta)    
        
        return means, posterior
    
    def compute_mode(self) -> torch.Tensor:
        """
        Compute mode of the distribution for each batch.
        
        Returns:
            Mode of distribution for each batch (batch_size, 3)
        """
        max_indices = torch.argmax(self.prior, dim=1)
        return self.grid_samples[max_indices]
    
    def _filter_out_of_bounds(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Filter out-of-bounds centroids.
        
        Args:
            centroids: Propagated centroids after motion step (N, 3)
            
        Returns:
            Boolean mask with in-bound centroids (N,)
        """
        mask_x = (centroids[..., 0] >= self.grid_bounds[0]) & (centroids[..., 0] <= self.grid_bounds[1])
        mask_y = (centroids[..., 1] >= self.grid_bounds[0]) & (centroids[..., 1] <= self.grid_bounds[1])
        mask = torch.logical_and(mask_x, mask_y)
        return mask
    
    def _compute_bin_indices(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Compute bin indices for each centroid in the grid.
        
        Args:
            centroids: Propagated centroids after motion step (N, 3)
            
        Returns:
            Bin index of each centroid (N, 3)
        """
        # Create bins for x, y, theta
        x_bins = torch.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[0] + 1, device=self.device)[:-1]  # Exclude last bin
        y_bins = torch.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[1] + 1, device=self.device)[:-1]  # Exclude last bin
        t_bins = torch.linspace(0, 2 * torch.pi, self.grid_size[2] + 1, device=self.device)[:-1]  # Exclude last bin
        # t_bins = torch.linspace(-torch.pi, torch.pi, self.grid_size[2] + 1, device=self.device)[:-1]  # Exclude last bin
        
        # Transform angle from [-pi, pi] to [0, 2pi]
        angles = centroids[:, :,  2].clone() % (2 * torch.pi)

        # Make tensors contiguous
        centroids = centroids.contiguous()
        x_bins = x_bins.contiguous()
        y_bins = y_bins.contiguous() 
        t_bins = t_bins.contiguous()
        
        # Compute bin indices
        idx = torch.searchsorted(x_bins, centroids[:, :, 0])
        idy = torch.searchsorted(y_bins, centroids[:, :, 1])
        # idt = torch.searchsorted(t_bins, centroids[:, :, 2])
        idt = torch.searchsorted(t_bins, angles)
        
        # Handle edge cases and convert to 0-indexed
        idx = torch.clamp(idx, 1, self.grid_size[0]) - 1
        idy = torch.clamp(idy, 1, self.grid_size[1]) - 1
        idt = torch.clamp(idt, 1, self.grid_size[2]) - 1
        
        return torch.stack([idx, idy, idt], dim=-1)
    
    def _compute_mean(self) -> torch.Tensor:
        """
        Compute expected value of the histogram bins for each batch.
        
        Returns:
            Mean value of the histogram bins (batch_size, 3)
        """
        # Expand grid_samples for batch processing
        # expanded_samples = self.grid_samples.unsqueeze(0)  # (1, N, 3)
        expanded_samples = self.grid_samples
        
        # Reshape prior for multiplication
        expanded_prior = self.prior.unsqueeze(-1)  # (batch_size, N, 1)
        
        # Element-wise multiplication and sum
        prod = expanded_samples * expanded_prior
        mean = torch.sum(prod, dim=1) # (batch_size, 3)
        
        return mean
    
    def neg_log_likelihood(self, pose: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """
        Evaluate posterior distribution of histogram filter.
        
        Args:
            pose: Poses at which to interpolate the SE2 Fourier transform (batch_size, 3)
            
        Returns:
            Negative log probability of distribution at given poses (batch_size,)
        """
        # Wrap pose angle to [0, 2pi]
        wrapped_pose = pose.clone()
        wrapped_pose[:, 2] = (wrapped_pose[:, 2] + 2 * torch.pi) % (2 * torch.pi)
        # pose[:, 2] = (pose[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize angles to [-pi, pi]
        
        # Find closest grid points for each batch
        neg_ll = torch.zeros(self.batch_size, device=self.device)
        
        for i in range(self.batch_size):
            diff = self.grid_samples[i] - wrapped_pose[i].unsqueeze(0)
            distances = torch.norm(diff, dim=1)
            idx = torch.argmin(distances)
            # Divide by cube's volume to obtain pdf
            ll = torch.log((prob[i, idx] / self.volume) + 1e-8)
            neg_ll[i] = -ll
        
        return torch.mean(neg_ll)

    def neg_log_likelihood_measurement(self, landmarks, measurement: torch.Tensor, energy_unnorm: torch.Tensor) -> torch.Tensor:
        
        # Find closest grid points for each batch
        neg_ll = torch.zeros(self.batch_size, device=self.device)
        
        if torch.sum(energy_unnorm) != 0:
            energy_unnorm /= torch.sum(energy_unnorm, dim=-1, keepdim=True)
        
        for i in range(self.batch_size):
            grid = torch.linalg.norm(landmarks - self.grid_samples[i, :, 0:2], dim=-1)
            diff = grid - measurement[i]
            # distances = torch.norm(diff, dim=1)
            idx = torch.argmin(diff)
            # Divide by cube's volume to obtain pdf
            ll = energy_unnorm[i, idx] / self.volume
            neg_ll[i] = -ll
        
        return torch.mean(neg_ll)
    
    def plot(self, batch_idx: int = 0, title: str = "") -> None:
        """
        Plot belief distribution for a specific batch.
        
        Args:
            batch_idx: Index of batch to plot
            title: Plot title
        """
        belief = self.prior[batch_idx].reshape(self.grid_size).cpu().numpy()
        
        xs = np.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[0], endpoint=False)
        ys = np.linspace(self.grid_bounds[0], self.grid_bounds[1], self.grid_size[1], endpoint=False)
        x, y = np.meshgrid(xs, ys, indexing='ij')
        
        plt.figure(figsize=(10, 8))
        # If I need to marginalise over theta 
        h = plt.contourf(x, y, belief.sum(-1))
        plt.axis('scaled')
        plt.colorbar()
        plt.title(f"{title} (Batch {batch_idx})")
        plt.show()


class BatchedBearingHF(BatchedRangeHF):
    def __init__(self, d_door2pose: float = 0.1, **kwargs):
        """
        Batched Histogram Filter for Bearing-based localization.
        
        Args:
            d_door2pose: Distance from door to pose
            **kwargs: Arguments passed to BatchedRangeHF
        """
        super().__init__(**kwargs)
        self.d_door2pose = d_door2pose
    
    def update(self,
               landmarks: torch.Tensor,
               map_mask: torch.Tensor,
               observations: torch.Tensor,
               observations_cov: torch.Tensor) -> torch.Tensor:
        """
        Update step for batched bearing HF.
        
        Args:
            landmarks: Location of each landmark (n_landmarks, 2)
            map_mask: Binary mask indicating traversable area (n_landmarks, N)
            observations: Bearing measurements (batch_size, m)
            observations_cov: Variance of each door (batch_size, n) or (n,)
            
        Returns:
            Mean pose for each batch (batch_size, 3)
        """
        # Ensure observations_cov is batched
        if observations_cov.dim() == 1:
            observations_cov = observations_cov.unsqueeze(0).expand(self.batch_size, -1)
        
        observations_std = torch.sqrt(observations_cov)
        
        # Compute measurement likelihood for each batch
        measurement_likelihood = torch.log(torch.tensor(1e-9, device=self.device)).expand(self.batch_size, self._N)
        
        # Calculate angles from grid points to landmarks
        grid_xy = self.grid_samples[:, :2]  # (N, 2)
        grid_theta = self.grid_samples[:, 2]  # (N)
        
        for batch_idx in range(self.batch_size):
            for i, obs in enumerate(observations[batch_idx]):
                # Compute angle differences for all landmarks and grid points
                diff = landmarks.unsqueeze(1) - grid_xy.unsqueeze(0)  # (n_landmarks, N, 2)
                
                # Calculate angles from grid points to landmarks
                angle = torch.atan2(diff[:, :, 1], diff[:, :, 0])
                
                # Adjust by grid point orientation
                grid_angle = ((grid_theta + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(0)
                angle = angle - grid_angle  # (n_landmarks, N)
                
                # Wrap angle to [-pi, pi]
                angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
                
                # Compute angle difference with observation
                diff_angle = obs - angle  # (n_landmarks, N)
                diff_angle = (diff_angle + torch.pi) % (2 * torch.pi) - torch.pi
                
                # Compute normal PDF for angle differences
                std_expanded = observations_std[batch_idx, i].unsqueeze(-1).expand(landmarks.shape[0], self._N)
                log_probs = -0.5 * (diff_angle / std_expanded)**2 - torch.log(std_expanded) - 0.5 * torch.log(2 * torch.pi)  # (n_landmarks, N)
                
                # Apply map mask
                log_probs = log_probs + torch.log(map_mask + 1e-8)
                
                # Take maximum over landmarks
                max_log_prob, _ = torch.max(log_probs, dim=0)  # (N,)
                
                # Update measurement likelihood
                measurement_likelihood[batch_idx] = torch.maximum(measurement_likelihood[batch_idx], max_log_prob)
        
        # Normalize measurement likelihood
        for batch_idx in range(self.batch_size):
            measurement_likelihood[batch_idx] -= torch.logsumexp(measurement_likelihood[batch_idx], dim=0)
        
        # Update and normalize posterior belief
        log_prior = torch.log(self.prior + 1e-8)
        log_posterior = log_prior + measurement_likelihood
        posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, dim=1, keepdim=True))
        
        # Update prior if there is probability mass
        self.prior = posterior
        
        # Compute mean for each batch
        means = self._compute_mean()
        
        return means