import torch
from scipy.special import i0
import math
import pdb
import numpy as np

class VonMissesDistribution:
    def __init__(self, mean, std, band_limit):
        self.mean = mean.numpy()
        self.std = std.numpy()
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



class MultimodalGaussianDistribution:
    def __init__(self, means, stds, n_modes,band_limit):
        self.means = means
        self.batch_size = means.shape[0]
        self.stds = stds
        self.n_modes = n_modes
        self.band_limit = band_limit
        self.weights = torch.ones(n_modes) / n_modes

    def energy(self):
        energy = torch.zeros((self.batch_size, self.band_limit))
        for i in range(self.n_modes):
            vmd = VonMissesDistribution(self.means[:,i], self.stds[i],self.band_limit)
            energy += self.weights[i]*vmd.density()
        return torch.log(energy).type(torch.FloatTensor)  
    
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