import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiablePF:
    def __init__(self,
                 prior_mu: torch.Tensor,
                 prior_cov: torch.Tensor,
                 n_particles: int = 100,
                 batch_size: int = 1,
                 grid_size = (100, 100, 36),
                 grid_bounds = (-0.5, 0.5),
                 alpha: float = 0.0,  # Soft resampling parameter
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        :param prior_mu: Prior pose as a torch tensor of dimension (B, 3) where B is batch size
        :param prior_cov: Covariance noise for the prior distribution (B, 3, 3) or (3, 3)
        :param n_particles: Number of particles to use
        :param batch_size: Number of independent particle filters to run in parallel
        :param alpha: Soft resampling parameter (0 = standard, hard resampling)
        :param device: Device to use for tensor operations
        """
        self._N = n_particles
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        
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
        self.log_weights = torch.ones((batch_size, n_particles), device=device) * (-math.log(n_particles))
        self.mode_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.mixture_std = 0.1  # Standard deviation for mixture likelihood
    
    def prediction(self, step: torch.Tensor, step_cov: torch.Tensor) -> torch.Tensor:
        """
        Prediction step PF
        :param step: motion step (relative displacement) of dimension (B, 3)
        :param step_cov: Covariance matrix of prediction step (B, 3, 3) or (3, 3)
        :return: Updated particles
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
        step_sample[:, :, 2] = (step_sample[:, :, 2] + math.pi) % (2 * math.pi) - math.pi
        
        # Apply step
        c = torch.cos(self.particles[:, :, 2])  # (B, N)
        s = torch.sin(self.particles[:, :, 2])  # (B, N)
        
        # Create a new tensor instead of in-place operation
        new_particles = torch.zeros_like(self.particles)
        
        # Update positions
        new_particles[:, :, 0] = self.particles[:, :, 0] + c * step_sample[:, :, 0] - s * step_sample[:, :, 1]
        new_particles[:, :, 1] = self.particles[:, :, 1] + s * step_sample[:, :, 0] + c * step_sample[:, :, 1]
        new_particles[:, :, 2] = self.particles[:, :, 2] + step_sample[:, :, 2]
        
        # Normalize angles
        new_particles[:, :, 2] = (new_particles[:, :, 2] + math.pi) % (2 * math.pi) - math.pi
        
        # Update particles
        self.particles = new_particles
        
        return self.particles
    
    def update_weights(self, observation_likelihood: torch.Tensor) -> torch.Tensor:
        """
        Update step for particle weights
        :param observation_likelihood: Log-likelihood for each particle (B, N)
        :return: Updated log weights (B, N)
        """
        # Update weights with the likelihood
        self.log_weights = self.log_weights + observation_likelihood
        
        # Normalize weights in log space
        log_sum = torch.logsumexp(self.log_weights, dim=1, keepdim=True)
        self.log_weights = self.log_weights - log_sum
        
        # Handle NaNs
        if torch.isnan(self.log_weights).any():
            print(f"Warning: Found nan in weights after normalization")
            self.log_weights = torch.nan_to_num(self.log_weights, nan=-math.log(self._N))
            log_sum = torch.logsumexp(self.log_weights, dim=1, keepdim=True)
            self.log_weights = self.log_weights - log_sum
        
        # Store mode index for each batch
        weights = torch.exp(self.log_weights)
        self.mode_index = torch.argmax(weights, dim=1)
        
        return self.log_weights
    
    def soft_resampling(self) -> None:
        """
        Performs differentiable soft resampling of particles
        """
        # Convert log weights to probabilities
        weights = torch.exp(self.log_weights)
        
        # Soft resampling - maintains gradient between old and new weights
        resample_prob = (1 - self.alpha) * weights + self.alpha/self._N
        new_weights = weights / resample_prob
        
        # Systematic resampling: samples evenly distributed over original particles
        base_indices = torch.linspace(0.0, 1.0 - 1.0/self._N, self._N, device=self.device)
        random_offsets = torch.rand(self.batch_size, 1, device=self.device) / self._N
        
        # Shape: (batch_size, num_particles)
        inds = random_offsets + base_indices[None, :]
        
        # Compute cumulative distribution
        cum_probs = torch.cumsum(resample_prob, dim=1)
        
        # Differentiable resampling using soft assignments
        # This is the key differentiable part - avoids hard indexing
        selection_weights = torch.zeros(
            (self.batch_size, self._N, self._N), 
            device=self.device
        )
        
        # Create a soft assignment matrix where each row represents how much each 
        # original particle contributes to a new particle
        for b in range(self.batch_size):
            for i in range(self._N):
                # For each target index, compute soft weights for each source particle
                # This replaces the hard indexing of searchsorted
                if i == 0:
                    left_prob = 0.0
                else:
                    left_prob = cum_probs[b, i-1]
                right_prob = cum_probs[b, i]
                
                # Compute overlap between each sample's bin and this particle's probability mass
                for j in range(self._N):
                    # Define the sample's bin
                    if j == 0:
                        bin_left = 0.0
                    else:
                        bin_left = inds[b, j-1]
                    bin_right = inds[b, j]
                    
                    # Calculate overlap
                    overlap_left = torch.max(left_prob, bin_left)
                    overlap_right = torch.min(right_prob, bin_right)
                    overlap = torch.clamp(overlap_right - overlap_left, min=0.0)
                    
                    # Assign weight proportional to overlap
                    if right_prob > left_prob:
                        selection_weights[b, j, i] = overlap / (right_prob - left_prob)
        
        # Use the soft assignment matrix to create new particles
        new_particles = torch.bmm(selection_weights, self.particles)
        
        # And new weights
        log_new_weights = torch.log(torch.bmm(
            selection_weights, 
            new_weights.unsqueeze(-1)
        ).squeeze(-1))
        
        # Normalize the new weights
        log_sum = torch.logsumexp(log_new_weights, dim=1, keepdim=True)
        log_new_weights = log_new_weights - log_sum
        
        # Update particles and weights
        self.particles = new_particles
        self.log_weights = log_new_weights
    
    def get_state_estimate(self) -> torch.Tensor:
        """
        Compute state estimate from particle distribution
        :return: Mean state estimate (B, 3)
        """
        # Convert log weights to probabilities
        weights = torch.exp(self.log_weights)
        
        # Compute weighted mean of particle states
        # For angles, we use circular statistics to handle wrap-around
        xy_mean = torch.sum(self.particles[:, :, :2] * weights.unsqueeze(-1), dim=1)
        
        # For angles, compute circular mean
        cos_theta = torch.sum(torch.cos(self.particles[:, :, 2]) * weights, dim=1, keepdim=True)
        sin_theta = torch.sum(torch.sin(self.particles[:, :, 2]) * weights, dim=1, keepdim=True)
        theta_mean = torch.atan2(sin_theta, cos_theta)
        
        # Combine results
        mean_state = torch.cat([xy_mean, theta_mean], dim=1)
        
        return mean_state
    
    def get_covariance(self) -> torch.Tensor:
        """
        Compute covariance of the particle distribution
        :return: Covariance matrix (B, 3, 3)
        """
        # Convert log weights to probabilities
        weights = torch.exp(self.log_weights)
        weights = weights.unsqueeze(-1)  # (B, N, 1)
        
        # Get mean state
        mean_state = self.get_state_estimate().unsqueeze(1)  # (B, 1, 3)
        
        # Compute difference between particles and mean
        # Taking care of circular difference for angles
        diff_xy = self.particles[:, :, :2] - mean_state[:, :, :2]
        diff_theta = torch.atan2(
            torch.sin(self.particles[:, :, 2] - mean_state[:, :, 2]),
            torch.cos(self.particles[:, :, 2] - mean_state[:, :, 2])
        )
        
        diff = torch.cat([diff_xy, diff_theta.unsqueeze(-1)], dim=2)  # (B, N, 3)
        
        # Compute weighted outer product
        diff_outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)  # (B, N, 3, 3)
        cov = torch.sum(diff_outer * weights.unsqueeze(-1), dim=1)  # (B, 3, 3)
        
        return cov
    
    def compute_mode(self) -> torch.Tensor:
        """
        Compute mode of the distribution
        :return: Mode of distribution (B, 3)
        """
        # Get the particles with highest weight for each batch
        return self.particles[torch.arange(self.batch_size, device=self.device), self.mode_index]

    # def differentiable_resampling(self) -> None:
    #     """
    #     Performs fully differentiable resampling using a soft assignment scheme
    #     """
    #     # Soft resampling with higher alpha for more gradient flow
    #     alpha_resample = max(0.1, self.alpha)  # Use at least some minimal alpha
        
    #     weights = torch.exp(self.log_weights)
        
    #     # Compute soft resampling probabilities
    #     resample_prob = (1 - alpha_resample) * weights + alpha_resample/self._N
        
    #     # Create a transportation plan between old and new particles
    #     # This is similar to optimal transport with entropic regularization
    #     transport_plan = torch.zeros((self.batch_size, self._N, self._N), device=self.device, dtype=torch.float64)
        
    #     # Initialize with uniform distribution
    #     target_dist = torch.ones((self.batch_size, self._N), device=self.device) / self._N
        
    #     # Simple Sinkhorn-like iterations to compute transport plan
    #     for _ in range(5):  # Usually 5-10 iterations are enough
    #         for b in range(self.batch_size):
    #             # Row normalization
    #             row_sum = transport_plan[b].sum(dim=1, keepdim=True)
    #             row_sum = torch.where(row_sum > 1e-12, row_sum, torch.ones_like(row_sum))
    #             transport_plan[b] = transport_plan[b] * target_dist[b].unsqueeze(1) / row_sum
                
    #             # Column normalization
    #             col_sum = transport_plan[b].sum(dim=0, keepdim=True)
    #             col_sum = torch.where(col_sum > 1e-12, col_sum, torch.ones_like(col_sum))
    #             transport_plan[b] = transport_plan[b] * resample_prob[b].unsqueeze(0) / col_sum
        
    #     # Transport particles according to plan
    #     new_particles = torch.bmm(transport_plan, self.particles)
        
    #     # Reset weights to uniform
    #     self.particles = new_particles
    #     self.log_weights = -torch.ones_like(self.log_weights) * math.log(self._N)

    # def differentiable_resampling(self) -> None:
    #     """
    #     Performs fully differentiable resampling using a soft assignment scheme
    #     """
    #     # Soft resampling with higher alpha for more gradient flow
    #     alpha_resample = max(0.1, self.alpha)  # Use at least some minimal alpha
        
    #     weights = torch.exp(self.log_weights)
        
    #     # Compute soft resampling probabilities
    #     resample_prob = (1 - alpha_resample) * weights + alpha_resample/self._N
        
    #     # Create a transportation plan between old and new particles
    #     # This is similar to optimal transport with entropic regularization
    #     transport_plan = torch.zeros((self.batch_size, self._N, self._N), device=self.device, dtype=torch.float64)
        
    #     # Initialize with uniform distribution
    #     target_dist = torch.ones((self.batch_size, self._N), device=self.device) / self._N
        
    #     # Simple Sinkhorn-like iterations to compute transport plan
    #     for _ in range(5):  # Usually 5-10 iterations are enough
    #         # Create a new tensor for the updated transport plan
    #         updated_transport_plan = transport_plan.clone()
            
    #         for b in range(self.batch_size):
    #             # Row normalization
    #             row_sum = transport_plan[b].sum(dim=1, keepdim=True)
    #             row_sum = torch.where(row_sum > 1e-12, row_sum, torch.ones_like(row_sum))
    #             row_normalized = transport_plan[b] * target_dist[b].unsqueeze(1) / row_sum
                
    #             # Column normalization
    #             col_sum = row_normalized.sum(dim=0, keepdim=True)
    #             col_sum = torch.where(col_sum > 1e-12, col_sum, torch.ones_like(col_sum))
    #             updated_transport_plan[b] = row_normalized * resample_prob[b].unsqueeze(0) / col_sum
            
    #         # Update the entire transport plan at once (out-of-place)
    #         transport_plan = updated_transport_plan
        
    #     # Transport particles according to plan
    #     new_particles = torch.bmm(transport_plan, self.particles)
        
    #     # Reset weights to uniform
    #     self.particles = new_particles
    #     self.log_weights = -torch.ones_like(self.log_weights) * math.log(self._N)

    def differentiable_resampling(self) -> None:
        """
        Performs fully differentiable resampling using a soft assignment scheme
        with memory-efficient implementation
        """
        # Soft resampling with higher alpha for more gradient flow
        alpha_resample = max(0.1, self.alpha)  # Use at least some minimal alpha
        
        # Compute soft resampling probabilities
        weights = torch.exp(self.log_weights)
        resample_prob = (1 - alpha_resample) * weights + alpha_resample/self._N
        
        # Create a transportation plan between old and new particles
        # Start with a small uniform value to ensure numerical stability
        transport_plan = torch.ones((self.batch_size, self._N, self._N), 
                                device=self.device, dtype=torch.float64) / (self._N * self._N)
        
        # Target distribution is uniform
        target_dist = torch.ones((self.batch_size, self._N), device=self.device, dtype=torch.float64) / self._N
        
        # Memory-efficient Sinkhorn iterations
        for _ in range(5):  # Usually 5-10 iterations are enough
            # Process the whole batch together to avoid redundant allocations
            
            # Row normalization - iterating over the full batch at once
            row_sum = transport_plan.sum(dim=2, keepdim=True)
            row_sum = torch.clamp_min(row_sum, 1e-12)  # Avoid division by zero
            transport_plan = transport_plan * (target_dist.unsqueeze(2) / row_sum)
            
            # Column normalization - iterating over the full batch at once
            col_sum = transport_plan.sum(dim=1, keepdim=True)
            col_sum = torch.clamp_min(col_sum, 1e-12)  # Avoid division by zero
            transport_plan = transport_plan * (resample_prob.unsqueeze(1) / col_sum)
        
        # Transport particles according to plan - matrix multiplication
        self.particles = torch.bmm(transport_plan, self.particles)
        
        # Reset weights to uniform
        self.log_weights = -torch.ones_like(self.log_weights) * math.log(self._N)
    
    def lightweight_soft_resampling(self) -> None:
        """Extremely lightweight soft resampling"""
        # Only slightly modify weights instead of full resampling
        alpha = 0.3  # Balance between current weights and uniform
        
        # Softly adjust weights toward uniform
        log_uniform = -torch.ones_like(self.log_weights) * math.log(self._N)
        self.log_weights = (1-alpha) * self.log_weights + alpha * log_uniform

    def mixture_likelihood(self, diffs, weights, reduce_mean=False):
        """
        Compute the negative log likelihood of y under a gaussian
        mixture model defined by a set of particles and their weights.
        
        Parameters
        ----------
        diffs : tensor
            difference between y and the states of the particles
        weights : tensor
            weights of the particles
        reduce_mean : bool, optional
            if true, return the mean likelihood loss over the complete tensor.
            The default is False.
            
        Returns
        -------
        likelihood : tensor
            the negative log likelihood
        """
        dim = diffs.size(-1)
        num = diffs.size(-2)
        
        # remove nans and infs and replace them with high values/zeros
        diffs = torch.where(torch.isfinite(diffs), diffs, torch.ones_like(diffs) * 1e5)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        covar = torch.ones(dim, dtype=torch.float32)
        for k in range(dim):
            covar[k] *= self.mixture_std
        covar = torch.diag(covar.square()).to(self.device)
        
        if len(diffs.shape) > 3:
            sl = diffs.size(1)
            diffs = diffs.reshape(self.batch_size, -1, num, dim, 1)
            covar = covar.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(self.batch_size, sl, num, -1, -1)
        else:
            sl = 1
            diffs = diffs.reshape(self.batch_size, num, dim, 1)
            covar = covar.unsqueeze(0).unsqueeze(0).expand(self.batch_size, num, -1, -1)
        
        # transfer to float64 for higher accuracy
        covar = covar.to(torch.float64)
        diffs = diffs.to(torch.float64)
        weights = weights.to(torch.float64)
        
        exponent = torch.matmul(torch.matmul(diffs.transpose(-2, -1), torch.inverse(covar)), diffs)
        exponent = exponent.reshape(self.batch_size, num)
        # print(exponent.shape)
        normalizer = torch.log(torch.det(covar)) + dim * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))
        # print(normalizer.shape)
        log_like = -0.5 * (exponent + normalizer)
        # print(log_like.shape)
        log_like = log_like.reshape(self.batch_size, num)
        log_like = torch.where(log_like >= -500, log_like, torch.ones_like(log_like) * -500)
        
        exp = torch.exp(log_like)
        # the per particle likelihoods are weighted and summed in the particle dimension
        weighted = weights * exp
        weighted = weighted.sum(dim=-1)
        
        # compute the negative logarithm and undo the bias
        likelihood = -(torch.log(torch.maximum(weighted, torch.tensor(1e-300, dtype=torch.float64))))
        
        if reduce_mean:
            likelihood = likelihood.mean()
        
        likelihood = likelihood.to(torch.float32)
        return likelihood

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

