import numpy as np
from scipy.stats import beta
import torch
import math

class BetaDistribution:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x):
        """Probability density function."""
        return beta.pdf(x, self.alpha, self.beta)/(2*math.pi)
    
    def density_over_grid(self,sample_size,batch_size):
        grid = np.tile(np.linspace(0, 1, sample_size +1)[:-1][None, :], (batch_size, 1))
        return torch.tensor(beta.pdf(grid, self.alpha, self.beta)).type(torch.FloatTensor)

    # def cdf(self, x):
    #     """Cumulative distribution function."""
    #     return beta.cdf(x, self.alpha, self.beta)

    def sample(self, size=1):
        """Generate random samples from the distribution."""
        return 2*math.pi*beta.rvs(self.alpha, self.beta, size=size)

    def mean(self):
        """Mean of the distribution."""
        return 2*math.pi*beta.mean(self.alpha, self.beta)

    def variance(self):
        """Variance of the distribution."""
        return beta.var(self.alpha, self.beta)
    
    def density_translated(self, theta_mean,sample_size):
        # theta_new = np.mod(self.mean() + theta_mean.squeeze(-1), 2 * math.pi)
        density_over_grid = self.density_over_grid(sample_size,theta_mean.shape[0])
        batch_size, seq_len = density_over_grid.shape
        shift_indices = (torch.arange(seq_len).unsqueeze(0) + theta_mean) % seq_len  # Wrap indices
        shift_indices = shift_indices.long()  # Ensure integer indices

        # Gather shifted values
        shifted_A = torch.gather(density_over_grid, dim=1, index=shift_indices)
        return shifted_A
    
    def negative_log_likelihood(self, x):
        """Negative log likelihood of the distribution."""
        return torch.tensor(-np.mean(np.log(self.pdf(x/(2*math.pi))))).type(torch.FloatTensor)