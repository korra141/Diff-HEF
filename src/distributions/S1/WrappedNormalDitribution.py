import torch
from scipy.special import i0
import math
import pdb
import numpy as np
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
from src.utils.metrics import absolute_error_s1

class VonMissesDistribution:
    def __init__(self, mean, std, band_limit):
        self.mean = mean
        self.std = std
        self.band_limit = band_limit
        self.batch_size = mean.shape[0]

    def density(self):
        """Probability density function"""
        grid = np.tile(np.linspace(0, 2*math.pi, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
        kappa = 1/(self.std **2)
        density = np.exp(kappa * np.cos(grid - self.mean)) / (2 * math.pi * i0(kappa))
        return torch.tensor(density).type(torch.FloatTensor)
    
    def density_local(self,range_theta):
        """Probability density function"""
        grid = np.tile(np.linspace(-range_theta/2, range_theta/2, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
        kappa = 1/(self.std **2)
        density = np.exp(kappa * np.cos(grid)) / (2 * math.pi * i0(kappa))
        return torch.tensor(density).type(torch.FloatTensor)
    
    def energy(self):
        grid = np.tile(np.linspace(0, 2*math.pi, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
        kappa = 1/(self.std **2)
        energy = kappa * np.cos(grid - self.mean)
        return torch.tensor(energy).type(torch.FloatTensor)
    
    def negative_loglikelihood(self,value):
        kappa = 1/(self.std **2)
        nll = - kappa * np.cos(value - self.mean) + np.log(2*math.pi*i0(kappa))
        nll[np.isinf(nll)] = 0
        return torch.mean(torch.tensor(nll).type(torch.FloatTensor))
    
    def density_value(self,value):
        # grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit+1)[:-1][None,:],(self.batch_size, 1))
        kappa = 1/(self.std **2)
        density = np.exp(kappa * np.cos(value - self.mean)) / (2 * math.pi * i0(kappa))
        return torch.tensor(density).type(torch.FloatTensor)

class WarpedNormalDistribution:
    def __init__(self, mean, cov, band_limit):
        self.mean = mean
        self.cov = cov
        self.band_limit = band_limit
        self.batch_size = mean.shape[0]

    def density(self):
        """Probability density function"""
        grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
        diff = absolute_error_s1(self.mean, grid)
        density = torch.exp(-0.5 * (diff)**2 / self.cov) / (torch.sqrt(2 * math.pi * self.cov))
        return density.type(torch.FloatTensor)
    
    # def density_local(self,range_theta):
    #     """Probability density function"""
    #     grid = np.tile(np.linspace(-range_theta/2, range_theta/2, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
    #     kappa = 1/(self.std **2)
    #     density = np.exp(kappa * np.cos(grid)) / (2 * math.pi * i0(kappa))
    #     return torch.tensor(density).type(torch.FloatTensor)
    
    def energy(self):
        grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit +1)[:-1][None, :], (self.batch_size, 1))
        diff = absolute_error_s1(self.mean, grid)
        energy = -0.5 * (diff)**2 / self.cov
        return energy.type(torch.FloatTensor)
    
    # def negative_loglikelihood(self,value):
    #     kappa = 1/(self.std **2)
    #     nll = - kappa * np.cos(value - self.mean) + np.log(2*math.pi*i0(kappa))
    #     nll[np.isinf(nll)] = 0
    #     return torch.mean(torch.tensor(nll).type(torch.FloatTensor))
    
    def density_value(self,value):
        # grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit+1)[:-1][None,:],(self.batch_size, 1))
        # kappa = 1/(self.std **2)
        diff = absolute_error_s1(self.mean, value)
        density = torch.exp(-0.5 * (diff)**2 / self.cov) / (torch.sqrt(2 * math.pi * self.cov))
        return density.type(torch.FloatTensor)
    
class VonMissesDistribution_torch:
    def __init__(self, mean, cov, band_limit):
        self.mean = mean
        self.cov = cov
        self.band_limit = band_limit
        self.batch_size = mean.shape[0]
    
    def safe_i0(self, kappa, log_i0=False):
      """
      Compute the modified Bessel function of the first kind, I_0(kappa),
      with a safe approach for large kappa values.
      """
      if torch.max(kappa) > 40:  # Handle large values
          if log_i0:
            i_0 = kappa - torch.log(2 * math.pi * kappa) / 2
          else:
            i_0 = torch.exp(kappa - torch.log(2 * math.pi * kappa) / 2)  
      else:
          if log_i0:
            i_0 = torch.log(torch.special.i0(kappa))
          else:
            i_0 = torch.special.i0(kappa)
      return i_0
    def energy(self):
        grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit+1)[:-1][None,:],(self.batch_size, 1))
        kappa = 1/(self.cov)
        energy = kappa * torch.cos(grid - self.mean)
        return energy.type(torch.FloatTensor)
    
    def negative_loglikelihood(self,value):
        kappa = 1/(self.cov)
        if torch.any(torch.isnan(self.safe_i0(kappa))):
            print("torch bessel is nan")
        return torch.mean(- kappa * torch.cos(value - self.mean) + torch.log(torch.tensor(2*math.pi)) + self.safe_i0(kappa, log_i0=True))
    
    def density(self):
        grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit+1)[:-1][None,:],(self.batch_size, 1))
        kappa = (1/(self.cov))
        mean = self.mean
        density = torch.exp(kappa * torch.cos(grid - mean)) / (2 * math.pi * self.safe_i0(kappa))
        temp = torch.sqrt(kappa/(2*math.pi))
        density = torch.where(torch.isinf(density), temp, density)
        density = torch.clamp(density, min=1e-10)
        density = torch.clamp(density, max=1e+10)
        density = density.nan_to_num_(nan=0.0)
        if (torch.isnan(density).any()):
            print("density is nan")
        return density
    
    def density_value(self,value):
        # grid = torch.tile(torch.linspace(0, 2*math.pi, self.band_limit+1)[:-1][None,:],(self.batch_size, 1))
        kappa = (1/(self.cov))
        density = torch.exp(kappa * torch.cos(value - self.mean)) / (2 * math.pi * self.safe_i0(kappa))
        temp = torch.sqrt(kappa/(2*math.pi))
        density = torch.where(torch.isinf(density), temp, density)
        density = torch.clamp(density, min=1e-10)
        density = torch.clamp(density, max=1e+10)
        density = density.nan_to_num_(nan=0.0)
        if (torch.isnan(density).any()):
            print("density is nan")
        return density


class MultimodalGaussianDistribution:
    def __init__(self, means, stds, weights, n_modes,band_limit):
        self.means = means.numpy()
        self.batch_size = means.shape[0]
        self.stds = stds.detach().numpy()
        self.n_modes = n_modes
        self.band_limit = band_limit
        self.weights = weights

    def energy(self):
        # pdb.set_trace()
        energy = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution(self.means[:,i], self.stds[i],self.band_limit)
            energy += self.weights[i]*vmd.density()
        return torch.log(torch.clamp(energy, min=1e-10)).type(torch.FloatTensor)
    
    def density(self):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution(self.means[:,i], self.stds[i],self.band_limit)
            density += self.weights[i]*vmd.density()
        return density.type(torch.FloatTensor)

    
    def density_local(self,range_theta):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution(self.means[:,i], self.stds[i],self.band_limit)
            density += self.weights[i]*vmd.density_local(range_theta)
        return density.type(torch.FloatTensor)  
    
    def negative_loglikelihood(self, value):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution(self.means[:,i], self.stds[i], self.band_limit)
            density += self.weights[i:i+1] * vmd.density_value(value)
        
        return torch.mean(-torch.log(torch.clamp(density,min=1e-10))).type(torch.FloatTensor)
    
class MultimodalGaussianDistribution_torch:
    def __init__(self, means, covs, pi, n_modes, band_limit):
        self.means = means
        self.batch_size = means.shape[0]
        # self.stds = sigma
        self.covs = covs
        self.n_modes = n_modes
        self.band_limit = band_limit
        # self.weights = torch.ones(n_modes) / n_modes
        self.weights = pi #[batch_size, n_modes]

    def energy(self):
        energy = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution_torch(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            energy += self.weights[:,i:i+1] * vmd.density()
        return torch.log(torch.clamp(energy,min=1e-10)).type(torch.FloatTensor)

    def density(self):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution_torch(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            density += self.weights[:,i:i+1] * vmd.density()
        return density.type(torch.FloatTensor)
    
    def negative_loglikelihood(self, value):
        density = torch.zeros((self.batch_size,1))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution_torch(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            density += self.weights[:,i:i+1] * vmd.density_value(value)
        return torch.mean(-torch.log(torch.sum(torch.clamp(density,min=1e-10), dim=-1))).type(torch.FloatTensor)
    
class MultimodalWrappedNormalDistribution:
    def __init__(self, means, covs, pi, n_modes, band_limit):
        self.means = means
        self.batch_size = means.shape[0]
        self.covs = covs
        self.n_modes = n_modes
        self.band_limit = band_limit
        self.weights = pi #[batch_size, n_modes]

    def energy(self):
        energy = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            energy += self.weights[:,i:i+1] * vmd.density()
        return torch.log(torch.clamp(energy,min=1e-10)).type(torch.FloatTensor)

    def density(self):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            density += self.weights[:,i:i+1] * vmd.density()
        return density.type(torch.FloatTensor)
    
    def negative_loglikelihood(self, value):
        density = torch.zeros((self.batch_size,1))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i:i+1], self.covs[:,i:i+1], self.band_limit)
            density += self.weights[:,i:i+1] * vmd.density_value(value)
        return torch.mean(-torch.log(torch.sum(torch.clamp(density,min=1e-10), dim=-1))).type(torch.FloatTensor)
    
    def energy_1cov(self):
        energy = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i], self.covs[i], self.band_limit)
            energy += self.weights[i] * vmd.density()
        return torch.log(torch.clamp(energy,min=1e-10)).type(torch.FloatTensor)

    def density_1cov(self):
        density = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i], self.covs[i], self.band_limit)
            density += self.weights[i] * vmd.density()
        return density.type(torch.FloatTensor)
    
    def negative_loglikelihood_1cov(self, value):
        density = torch.zeros((self.batch_size,1))
        for i in range(self.n_modes):
            vmd = WarpedNormalDistribution(self.means[:, i], self.covs[i], self.band_limit)
            density += self.weights[i] * vmd.density_value(value)
        return torch.mean(-torch.log(torch.sum(torch.clamp(density,min=1e-10), dim=-1))).type(torch.FloatTensor)