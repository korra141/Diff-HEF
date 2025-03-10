import torch
import math
import numpy as np
# from scipy.stats import norm
import pdb

class HarmonicExponentialDistribution:
    def __init__(self, bandwidth,range_theta=None):
        self.range_theta = range_theta if range_theta is not None else 2 * math.pi
        if range_theta is None:
            self.local_grid = False
        else:
            self.local_grid = True
        self.grid_size = bandwidth
    
    def loss_regularisation_norm(self,lambda_,energy,value,center=None,scaling_factor=1):
        if not self.local_grid and center is None:
            NLL = self.negative_log_likelihood(energy, value, mean=False)
        else:
            NLL = self.negative_log_likelihood_local(energy,value,center,mean=False)
        grid_norm = torch.norm(energy, p=1, dim=-1, keepdim=False, out=None)
        max = torch.max(grid_norm)  
        grid_norm = (grid_norm/max)*scaling_factor
        return torch.mean(NLL + lambda_ * grid_norm)

    def negative_log_likelihood(self,energy, measurements, mean=True):
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
        k_values = k_values.unsqueeze(0).unsqueeze(-1).to(measurements.device) # [1, num_samples, 1]
        value = measurements.unsqueeze(1)  # [batch_size, 1, 1]
        exponential_term = torch.exp(1j * k_values * value)  # [batch_size, num_samples, 1]

        inverse_transform = (eta.unsqueeze(-1) * exponential_term).sum(dim=1).real # [batch_size, 1]
        if mean:
            return torch.mean(-inverse_transform/self.grid_size + ln_z_, axis=0)
        else:
            return -inverse_transform/self.grid_size + ln_z_
    
    def negative_log_likelihood_local(self,energy, measurements, center,mean=True):
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
        value = measurements - center # [batch_size, 1]
        value = value + self.range_theta/2
        # mean = mean.unsqueeze(-1) # [batch_size, 1,1]
        exponential_term = torch.exp((1j * 2*math.pi* k_values * value)/self.range_theta)  # [batch_size, num_samples]
        # exponential_term_mean = torch.exp((1j * 2*math.pi * k_values * mean)/range_theta)  # [batch_size, num_samples]

        inverse_transform = (eta * exponential_term).sum(dim=-1).real # [batch_size]
        if mean:
            return torch.mean(-inverse_transform/self.grid_size + ln_z_)
        else:
            return -inverse_transform/self.grid_size + ln_z_
    
    def normalization_constant(self,energy):
        maximum = torch.max(energy, dim=-1).values.unsqueeze(-1)
        moments = torch.fft.fft(torch.exp(energy - maximum), dim=-1)
        ln_z_ = torch.log(self.range_theta * moments[:, 0] / self.grid_size).real.unsqueeze(-1) + maximum
        return ln_z_
    
    def mode(self,predicted_density,ground_truth, n_modes=None):
        # Step 1: Get the maximum values and their indices along the last dimension (dim=-1)
          # max over columns, shape (batch_size,)
        if n_modes is not None:
            max_vals, max_indices = torch.topk(predicted_density, n_modes, dim=-1)
        else:
            max_vals, max_indices = torch.max(predicted_density, dim=-1)
        # Step 2: Get the indices of the maximum values along the last dimension (dim=-1)
        if self.local_grid:
            poses_mode = (self.range_theta * max_indices / self.grid_size) + ground_truth  - self.range_theta/2
        else:
            poses_mode = (self.range_theta  * max_indices / self.grid_size)

        return poses_mode.unsqueeze(-1)

    def mean(self,predicted_density):
        batch_size = predicted_density.shape[0]
        grid = torch.tile(torch.linspace(0, 2*math.pi, self.grid_size+1)[:-1][None,:],(batch_size, 1))
        diff = torch.ones_like(grid) * (2 * math.pi / self.grid_size)
        mean = torch.sum(predicted_density* grid * diff, dim=-1)
        return mean.unsqueeze(-1)