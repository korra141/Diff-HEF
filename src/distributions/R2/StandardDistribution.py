import numpy as np
import torch
import math
import pdb

class GaussianDistribution:
    def __init__(self, mean, std, x_range, y_range, band_limit):
        self.mean = mean
        self.std = std
        self.band_limit = band_limit
        self.x_range = x_range
        self.y_range = y_range

    def negative_log_likelihood(self, value):
        # x, y = np.meshgrid(self.x_range, self.y_range)
        exponent = -((value[:,0:1] - self.mean[:, 0:1])**2 / (2 * self.std**2)) - ((value[:,1:2] - self.mean[:, 1:2])**2 / (2 * self.std**2))
        normalization = torch.tensor(2*math.pi*(self.std**2))
        return torch.mean(-exponent + torch.log(normalization))


    def density_over_grid(self):

        n_traj = self.mean.shape[0]
        # Create a meshgrid for the x and y coordinates
        x = torch.linspace(self.x_range[0], self.x_range[1], self.band_limit[0]+1)[:-1]
        y = torch.linspace(self.y_range[0], self.y_range[1], self.band_limit[1]+1)[:-1]
        x, y = torch.meshgrid(x, y)

        # Convert to numpy arrays for compatibility (if needed)
        x = torch.tile(x.unsqueeze(0),(n_traj,1,1))  # Add batch dimension
        y = torch.tile(y.unsqueeze(0),(n_traj,1,1))

        # Calculate the true unnormalized density for each ground truth point in the batch
        exponent =  -((x - self.mean[:, 0:1, None])**2 / (2 * self.std**2)) - ((y - self.mean[:, 1:2, None])**2 / (2 * self.std**2))
        normalization = torch.tensor(2*math.pi*(self.std**2))

        return torch.exp(exponent)/normalization
    def energy_over_grid(self):

        n_traj = self.mean.shape[0]
        # Create a meshgrid for the x and y coordinates
        x = torch.linspace(self.x_range[0], self.x_range[1], self.band_limit[0]+1)[:-1]
        y = torch.linspace(self.y_range[0], self.y_range[1], self.band_limit[1]+1)[:-1]
        x, y = torch.meshgrid(x, y)

        # Convert to numpy arrays for compatibility (if needed)
        x = torch.tile(x.unsqueeze(0),(n_traj,1,1))  # Add batch dimension
        y = torch.tile(y.unsqueeze(0),(n_traj,1,1))

        # Calculate the true unnormalized density for each ground truth point in the batch
        exponent =  -((x - self.mean[:, 0:1, None])**2 / (2 * self.std**2)) - ((y - self.mean[:, 1:2, None])**2 / (2 * self.std**2))
        normalization = torch.tensor(2*math.pi*(self.std**2))

        return exponent - torch.log(normalization)
    

class MultiModalGaussianDistribution:
    def __init__(self, mean, std, x_range, y_range, band_limit, n_modes):
        self.mean = mean # Shape: (n_traj, n_modes, 2)
        self.std = std # Shape: (n_modes)
        self.x_range = x_range
        self.y_range = y_range
        self.band_limit = band_limit
        self.n_modes = n_modes
        self.weight = torch.ones(n_modes) / n_modes


    def negative_log_likelihood(self, value):
        n_samples = value.shape[0]
        pdf_values = torch.zeros((n_samples,1))

        for i in range(self.n_modes):
            exponent = -((value[:,0:1] - self.mean[:, i, 0:1])**2 / (2 * self.std[i]**2)) - ((value[:, 1:2] - self.mean[:,i, 1:2])**2 / (2 * self.std[i]**2))
            normalization = 2 * math.pi * (self.std[i]**2)
            pdf_values += self.weight[i] * torch.exp(exponent) / normalization

        return torch.mean(-torch.log(pdf_values)) # Shape: (n_traj))


    def density_over_grid(self):

        n_traj = self.mean.shape[0]
        # Create a meshgrid for the x and y coordinates
        x = torch.linspace(self.x_range[0], self.x_range[1], self.band_limit[0]+1)[:-1]
        y = torch.linspace(self.y_range[0], self.y_range[1], self.band_limit[1]+1)[:-1]
        x, y = torch.meshgrid(x, y)

        # Convert to numpy arrays for compatibility (if needed)
        x = torch.tile(x.unsqueeze(0),(n_traj,1,1))  # Add batch dimension
        y = torch.tile(y.unsqueeze(0),(n_traj,1,1))

        pdf_values = torch.zeros_like((x))

        for i in range(self.n_modes):
            exponent = -((x - self.mean[:, i, 0:1, None])**2 / (2 * self.std[i]**2)) - ((y - self.mean[:,i, 1:2, None])**2 / (2 * self.std[i]**2))
            normalization = 2 * math.pi * (self.std[i]**2)
            pdf_values += self.weight[i] * torch.exp(exponent) / normalization

        return pdf_values


