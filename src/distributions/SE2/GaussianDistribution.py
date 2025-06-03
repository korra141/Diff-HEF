import torch
import pdb

class MultiModalGaussianSE2:
    def __init__(self, mus, covs, spatial_grid_size):
        """
        Initialize the multimodal Gaussian distribution in SE(2) with equal weights.
        
        Args:
            mus (list of torch.Tensor): List of mean vectors, each of shape (3,).
            covs (list of torch.Tensor): List of covariance matrices, each of shape (3, 3).
            spatial_grid_size (tuple): Size of the spatial grid.
        """
        self.distributions = [GaussianSE2(mu, cov, spatial_grid_size) for mu, cov in zip(mus, covs)]
        self.n_components = len(self.distributions)
        self.spatial_grid_size = spatial_grid_size

    def energy(self, x):
        """
        Compute the energy (log of unnormalized density) for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Energy values for each input pose, reshaped to spatial grid size.
        """
        log_weight = torch.log(1.0 / self.n_components)
        energies = torch.stack([torch.exp(log_weight + component.energy(x)) for component in self.distributions], dim=-1)
        energy = torch.log(torch.sum(energies) + 1e-8)
        return energy

    def density(self, x):
        """
        Compute the density for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Density values for each input pose, reshaped to spatial grid size.
        """
        weight = 1.0 / self.n_components
        densities = torch.stack([component.density(x) for component in self.distributions], dim=-1)
        total_density = torch.sum(weight * densities, dim=-1)
        return total_density

    def negative_log_likelihood(self, x):
        """
        Compute the negative log likelihood for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Negative log likelihood values for each input pose, reshaped to spatial grid size.
        """
        log_weight = torch.log(1.0 / self.n_components)
        nlls = torch.stack([torch.exp(log_weight + component.negative_log_likelihood(x)) for component in self.distributions], dim=-1)
        nll = torch.log(torch.sum(nlls) + 1e-8)
        return nll
        
class GaussianSE2:
    def __init__(self, mu, cov, spatial_grid_size):
        """
        Initialize the Gaussian distribution in SE(2).
        
        Args:
            mu (torch.Tensor): Mean vector of shape (3,).
            cov (torch.Tensor): Covariance matrix of shape (3, 3).
            spatial_grid_size (tuple): Size of the spatial grid.
        """
        self.mu = mu
        self.inv_cov = torch.linalg.inv(cov)
        self.det_cov = torch.linalg.det(cov)
        self.normalizing_constant = torch.sqrt((2 * torch.pi) ** 3 * self.det_cov)
        self.spatial_grid_size = spatial_grid_size

    def energy(self, x):
        """
        Compute the energy (log of unnormalized density) for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Energy values for each input pose, reshaped to spatial grid size.
        """
        # Compute difference from the mean
        diff = x - self.mu

        # Wrap the angular difference to [-pi, pi]
        diff[:, :, 2] = (diff[:,:, 2] + torch.pi) % (2*torch.pi) - torch.pi

        diff = diff.unsqueeze(2)

        # Compute the energy
        # quadratic_form = torch.sum((diff @ self.inv_cov) * diff, dim=-1)
        quadratic_form = diff @ self.inv_cov @ diff.transpose(-1, -2)
        energy = -0.5 * quadratic_form.squeeze(-1).squeeze(-1)  # Remove the last two dimensions 
        normalisation_constant = torch.tile(- torch.log(self.normalizing_constant), energy.shape)
        # + normalisation_constant
        # Reshape to spatial grid size
        return energy + normalisation_constant

    def density(self, x):
        """
        Compute the density for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Density values for each input pose, reshaped to spatial grid size.
        """
        # Compute difference from the mean
        diff = x - self.mu.unsqueeze(1)
        # diff = x - self.mu

        # Wrap the angular difference to [-pi, pi]
        diff[:, :, 2] = (diff[:,:, 2] + torch.pi) % (2*torch.pi) - torch.pi

        # Compute the density
        # quadratic_form = torch.sum((diff @ self.inv_cov) * diff, dim=-1)
        diff = diff.unsqueeze(2)

        # Compute the energy
        # quadratic_form = torch.sum((diff @ self.inv_cov) * diff, dim=-1)
        quadratic_form = diff @ self.inv_cov @ diff.transpose(-1, -2)
        energy = -0.5 * quadratic_form.squeeze(-1).squeeze(-1)  # Remove the last two dimensions 
        density = torch.exp(energy)/self.normalizing_constant

        # Reshape to spatial grid size
        return density

    def negative_log_likelihood(self, x):
        """
        Compute the negative log likelihood for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Negative log likelihood values for each input pose, reshaped to spatial grid size.
        """
        unnormalised_nll = -self.energy(x)
        normalisation_constant = torch.tile(torch.log(self.normalizing_constant),unnormalised_nll.shape)
        # + normalisation_constant
        return unnormalised_nll 