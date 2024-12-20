import numpy as np
import torch
import math
import pdb

class GaussianDistribution:
    def __init__(self, mean, cov, band_limit ,x_range=None, y_range=None,range_x_diff=None,range_y_diff=None):
        self.mean = mean
        self.cov = cov
        self.band_limit = band_limit
        if x_range is not None and y_range is not None:    
            self.x_range = x_range
            self.y_range = y_range
        if range_x_diff is not None and range_x_diff is not None:
            self.range_x_diff = range_x_diff
            self.range_y_diff = range_y_diff
        else:
            self.x_range_diff = x_range[1] - x_range[0]
            self.y_range_diff = y_range[1] - y_range[0]

    def negative_log_likelihood(self, value):
        # x, y = np.meshgrid(self.x_range, self.y_range)
        exponent = -((value[:,0] - self.mean[:, 0])**2 / (2 * self.cov[:,0])) - ((value[:,1] - self.mean[:, 1])**2 / (2 * self.cov[:,1]))
        sqrt_det_cov = torch.sqrt(self.cov[:, 0] * self.cov[:, 1])
        normalization = torch.tensor(2*math.pi*(sqrt_det_cov))
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
        exponent =  -((x - self.mean[:, 0:1, None])**2 / (2 * self.cov[:,0:1,None])) - ((y - self.mean[:, 1:2, None])**2 / (2 * self.cov[:,1:2,None]))
        sqrt_det_cov = torch.sqrt(self.cov[:, 0] * self.cov[:, 1])
        normalization = torch.tensor(2*math.pi*(sqrt_det_cov))[:,None,None]

        return torch.exp(exponent)/normalization
    
    def density_over_local_grid(self):
        n_traj,dim = self.mean.shape
        # Create a meshgrid for the x and y coordinates
        x = torch.linspace(-self.range_x_diff/2 ,self.range_x_diff/2, self.band_limit[0]+1)[:-1]
        y = torch.linspace(-self.range_y_diff/2 ,self.range_y_diff/2, self.band_limit[1]+1)[:-1]
        x, y = torch.meshgrid(x, y)

        # Convert to numpy arrays for compatibility (if needed)
        x = torch.tile(x.unsqueeze(0),(n_traj,1,1))  # Add batch dimension
        y = torch.tile(y.unsqueeze(0),(n_traj,1,1))

        # Calculate the true unnormalized density for each ground truth point in the batch
        exponent =  -((x)**2 / (2 * self.cov[:,0:1,None] )) - ((y)**2 / (2 * self.cov[:,1:2,None]))
        sqrt_det_cov = torch.sqrt(self.cov[:, 0] * self.cov[:, 1])
        normalization = torch.tensor(2*math.pi*(sqrt_det_cov))[:,None,None]
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
        exponent =  -((x - self.mean[:, 0:1, None])**2 / (2 * self.cov[:,0:1,None])) - ((y - self.mean[:, 1:2, None])**2 / (2 * self.cov[:,1:2,None]))
        sqrt_det_cov = torch.sqrt(self.cov[:, 0] * self.cov[:, 1])
        normalization = torch.tensor(2*math.pi*(sqrt_det_cov))[:,None,None]

        return exponent - torch.log(normalization)
    

class MultiModalGaussianDistribution:
    def __init__(self, mean, std, band_limit, n_modes,x_range=None, y_range=None,range_x_diff=None,range_y_diff=None):
        self.mean = mean # Shape: (n_traj, n_modes, 2)
        self.std = std # Shape: (n_modes)
        self.x_range = x_range
        self.y_range = y_range
        self.band_limit = band_limit
        self.n_modes = n_modes
        self.weight = torch.ones(n_modes) / n_modes
        if x_range is not None and y_range is not None:    
            self.x_range = x_range
            self.y_range = y_range
        if range_x_diff is not None and range_y_diff is not None:
            self.range_x_local_diff = range_x_diff
            self.range_y_local_diff = range_y_diff
        else:
            self.range_x_local_diff = x_range[1] - x_range[0]
            self.range_y_local_diff = y_range[1] - y_range[0]


    def negative_log_likelihood(self, value):
        n_samples = value.shape[0]
        pdf_values = torch.zeros((n_samples,1))

        for i in range(self.n_modes):
            exponent = -((value[:,0:1] - self.mean[:, i, 0:1])**2 / (2 * self.std[i]**2)) - ((value[:, 1:2] - self.mean[:,i, 1:2])**2 / (2 * self.std[i]**2))
            normalization = 2 * math.pi * (self.std[i]**2)
            pdf_values += self.weight[i] * torch.exp(exponent) / normalization

        return torch.mean(-torch.log(pdf_values)) # Shape: (n_traj))
    
    def density_over_local_grid(self):

        weight = torch.ones(self.n_modes) / self.n_modes
        n_traj = self.mean.shape[0]
        # Create a meshgrid for the x and y coordinates
        x = torch.linspace(-self.range_x_local_diff/2 ,self.range_x_local_diff/2, self.band_limit[0]+1)[:-1]
        y = torch.linspace(-self.range_y_local_diff/2 ,self.range_y_local_diff/2, self.band_limit[1]+1)[:-1]
        x, y = torch.meshgrid(x, y)

        # Convert to numpy arrays for compatibility (if needed)
        x = torch.tile(x.unsqueeze(0),(n_traj,1,1))  # Add batch dimension
        y = torch.tile(y.unsqueeze(0),(n_traj,1,1))

        pdf_values = torch.zeros_like((x))

        for i in range(self.n_modes):
            exponent = -((x)**2 / (2 * self.std[i]**2)) - ((y)**2 / (2 * self.std[i]**2))
            normalization = 2 * math.pi * (self.std[i]**2)
            pdf_values += weight[i] * torch.exp(exponent) / normalization

        return pdf_values


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


