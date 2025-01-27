import torch

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
        diff[:, :, 2] = (diff[:,:, 2] + torch.pi) % torch.pi

        # Compute the energy
        quadratic_form = torch.sum((diff @ self.inv_cov) * diff, dim=-1)
        energy = -0.5 * quadratic_form 
#         - torch.log(self.normalizing_constant)

        # Reshape to spatial grid size
        return energy

    def density(self, x):
        """
        Compute the density for given poses in SE(2).
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3), where each row is a pose (x, y, theta).
        
        Returns:
            torch.Tensor: Density values for each input pose, reshaped to spatial grid size.
        """

        # Compute difference from the mean
        diff = x - self.mu

        # Wrap the angular difference to [-pi, pi]
        diff[:, :, 2] = (diff[:, : , 2] + torch.pi) % (torch.pi)

        # Compute the density
        quadratic_form = torch.sum((diff @ self.inv_cov) * diff, dim=-1)
        density = torch.exp(-0.5 * quadratic_form)/self.normalizing_constant

        # Reshape to spatial grid size
        return density
