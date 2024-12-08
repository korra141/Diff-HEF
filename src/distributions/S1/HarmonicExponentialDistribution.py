import torch
import math
import numpy as np
# from scipy.stats import norm
import pdb

class HarmonicExponentialDistribution:
    def __init__(self, bandwidth,range_theta=None):
        self.range_theta = range_theta if range_theta is not None else 2 * math.pi
        if range_theta is None:
            self.local_grid = True
        else:
            self.local_grid = False
        self.grid_size = bandwidth

    def negative_log_likelihood(self,energy, measurements):
        """Computes the loss function for the circular motion model.

        This function calculates the loss based on the negative log likelihood of the predicted distribution with the ground truth noisy measurements value
        Args:
            mu: The mean angle of the predicted distribution.
            cov: The covariance of the predicted distribution.
            measurements: The observed measurements.
            grid_size: The number of grid points to use for calculating the energy.

        Returns:
            ln_z_: The computed loss value.
        """
        # energy = compute_energy(mu, cov, grid_size)
        eta = torch.fft.fftshift(torch.fft.fft(energy, dim=-1), dim=-1)
        ln_z_ =  self.normalization_constant(energy)

        # taking inverse FFT over a set of frequencies ranging
        k_values = torch.arange(self.grid_size) - self.grid_size / 2
        k_values = k_values.unsqueeze(0).unsqueeze(-1)  # [1, num_samples, 1]
        value = measurements.unsqueeze(1)  # [batch_size, 1, 1]
        exponential_term = torch.exp(1j * k_values * value)  # [batch_size, num_samples, 1]

        inverse_transform = (eta.unsqueeze(-1) * exponential_term).sum(dim=1).real # [batch_size, 1]
        return torch.mean(-inverse_transform/self.grid_size + ln_z_, axis=0)
    
    def negative_log_likelihood_local(self,energy, measurements, mean):
        """Computes the loss function for the circular motion model.

        This function calculates the loss based on the negative log likelihood of the predicted distribution with the ground truth noisy measurements value
        Args:
            mu: The mean angle of the predicted distribution.
            cov: The covariance of the predicted distribution.
            measurements: The observed measurements.
            grid_size: The number of grid points to use for calculating the energy.

        Returns:
            ln_z_: The computed loss value.
        """

        eta = torch.fft.fftshift(torch.fft.fft(energy, dim=-1),dim=-1)

        ln_z_ = self.normalization_constant(energy)
        # taking inverse FFT over a set of frequencies ranging
        k_values = torch.arange(self.grid_size) - self.grid_size / 2
        k_values = k_values.unsqueeze(0) # [1, num_samples]
        value = measurements - mean # [batch_size, 1]
        value = value + self.range_theta/2
        # mean = mean.unsqueeze(-1) # [batch_size, 1,1]
        exponential_term = torch.exp((1j * 2*math.pi* k_values * value)/self.range_theta)  # [batch_size, num_samples]
        # exponential_term_mean = torch.exp((1j * 2*math.pi * k_values * mean)/range_theta)  # [batch_size, num_samples]

        inverse_transform = (eta * exponential_term).sum(dim=-1).real # [batch_size]
        return torch.mean(-inverse_transform/self.grid_size + ln_z_)
    
    def normalization_constant(self,energy):
        maximum = torch.max(energy, dim=-1).values.unsqueeze(-1)
        moments = torch.fft.fft(torch.exp(energy - maximum), dim=-1)
        ln_z_ = torch.log(self.range_theta * moments[:, 0] / self.grid_size).real.unsqueeze(-1) + maximum
        return ln_z_
    
    def mode(self,predicted_density,ground_truth):
        # Step 1: Get the maximum values and their indices along the last dimension (dim=-1)
        max_vals, max_indices = torch.max(predicted_density, dim=-1)  # max over columns, shape (batch_size,)

        # Step 2: Get the indices of the maximum values along the last dimension (dim=-1)
        if self.local_grid:
            poses_mode = (self.range_theta * max_indices / self.grid_size) + ground_truth  - self.range_theta/2
        else:
            poses_mode = (self.range_theta  * max_indices / self.grid_size)

        return poses_mode
