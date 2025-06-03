"""
Distributions defined over the SE(2) motion group.
"""
from typing import Tuple, Type, List
from abc import ABCMeta
import torch
from torch.distributions import MultivariateNormal
import pdb
import math

from src.distributions.SE2.distribution_base import HarmonicExponentialDistribution
# from distributions.SE2.distribution_base import HarmonicExponentialDistribution


class SE2(HarmonicExponentialDistribution, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO: Get rid of this once SE2 real fft is implemented.
    def from_samples(self) -> None:
        """
        Set up the distribution from samples
        :return : none
        """
        # Compute energy of samples
        energy = self.compute_energy(self.samples)
        _, _, _, _, _, self.eta = self.fft.analyze(energy)
        # This seems redundant, but as there is a loss of information in FFT analyze due to cartesian to polar
        # interpolation, the normalization constant is computed wrt to the energy synthesize by the eta params and not
        # by the one originally used as input. Therefore, normalizing the "original" energy starts giving bad results
        self.energy, _, _, _, _, _ = self.fft.synthesize(self.eta)

    @classmethod
    def product(cls, dist1: Type['SE2'], dist2: Type['SE2']) -> Type['SE2']:
        """
        Product of two distribution and update canonical parameters for S1 group
        :param dist1: first distribution
        :param dist2: second distribution to multiply with
        :return: S1 distribution
        """
        eta = dist1.eta + dist2.eta
        # pdb.set_trace()
        return cls.from_eta(eta, dist1.fft)

    @staticmethod
    def mul(fh1, fh2):
        assert fh1.shape == fh2.shape

        # The axes of fh are (r, p, q)
        # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
        # outside the range stored.
        # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
        # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
        p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
        q0 = fh1.shape[2] // 2

        # The lower and upper bound of the p-range
        a = p0 - q0
        b = int(p0 + torch.ceil(torch.tensor(fh2.shape[2] / 2.)).item())

        # PyTorch equivalent of np.einsum('rpn,rnn->rpn', fh1, fh2[:, a:b, :])
        fh12 = torch.bmm(fh1, fh2[:, a:b, :])
        return fh12

    # @staticmethod
    # def mulT(fh1, fh2):
    #     assert fh1.shape == fh2.shape

    #     # The axes of fh are (r, p, q) -> (p, n, m)
    #     # For each r, we multiply the infinite dimensional matrices indexed by (p, q), assuming the values are zero
    #     # outside the range stored.
    #     # Thus, the p-axis of the second array fh2 must be truncated at both sides so that we can compute fh1.dot(fh2),
    #     # and so that the 0-frequency q-component of fh1 lines up with the zero-fruency p-component of fh2.
    #     p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
    #     q0 = fh1.shape[2] // 2

    #     # The lower and upper bound of the p-range
    #     a = p0 - q0
    #     b = int(p0 + torch.ceil(torch.tensor(fh2.shape[2] / 2.)).item())

    #     fh12 = torch.zeros_like(fh1)
    #     for i in range(fh1.shape[0]):
    #         fh12[i, :, :] = torch.mm(fh1[i, :, :], fh2[i, :, :].T)[:, a:b]

    #     return fh12

    # @staticmethod
    # def mulT(fh1, fh2):
    #     assert fh1.shape == fh2.shape

    #     batch_size, p, q, r = fh1.shape

    #     p0 = q // 2  # Indices of the zero frequency component
    #     q0 = r // 2

    #     # The lower and upper bound of the p-range
    #     a = p0 - q0
    #     b = int(p0 + torch.ceil(torch.tensor(r / 2.)))

    #     fh12 = torch.zeros_like(fh1)  # Initialize output tensor
    #     for j in range(batch_size):
    #         for i in range(p):
    #             fh12[j, i, :, :] = torch.matmul(fh1[j, i, :, :], fh2[j, i, :, :].T)[:, a:b]

    #     return fh12

    def mulT(fh1, fh2):
        assert fh1.shape == fh2.shape

        batch_size, r, p, q = fh1.shape

        p0 = p // 2  # Indices of the zero frequency component
        q0 = q // 2

        # The lower and upper bound of the p-range
        a = p0 - q0
        b = int(p0 + int(torch.ceil(torch.tensor(q / 2.))))

        # fh12 = torch.zeros_like(fh1)  # Initialize output tensor
        results = []
        for j in range(batch_size):
            batch_result= []
            for i in range(r):
                # fh12[j, i, :, :] = torch.matmul(fh1[j, i, :, :], fh2[j, i, :, :].T)[:, a:b]
                prod = fh1[j,i] @ fh2[j,i].T 
                # prod = fh1[j, i, :, :] @ fh2[j,i, a:b, :]
                batch_result.append(prod[:, a:b])
                # batch_result.append(prod)
            batch_result = torch.stack(batch_result, dim=0)
            results.append(batch_result)
        fh12 = torch.stack(results, dim=0)

        return fh12


    @classmethod
    def convolve(cls, dist1: Type['SE2'], dist2: Type['SE2']) -> Type['SE2']:
        """
        Convolution of two distribution and update canonical parameters in log
        space for SE2 group
        :param dist1: first distribution
        :param dist2: second distribution to convolve with
        :return: SE2 distribution
        """
        # pdb.set_trace()
        M = cls.mulT(dist2.M, dist1.M)
        return cls.from_M(M, dist1.fft)

    def normalize(self) -> None:
        """
        Updated moments, energy, probability, log partition constant and compute
        Ms for SE2 group
        :return: none
        """
        _, _ = self.compute_moments_lnz(self.eta, update=True)
        prob = torch.exp(self.energy - self.l_n_z) + 1e-8
        _, _, _, _, _, self.M = self.fft.analyze(prob)
        prob, _, _, _, _, _ = self.fft.synthesize(self.M)
        self.prob = torch.where(prob.real > 0, prob.real, 1e-8)

    def compute_eta(self) -> None:
        """
        Compute eta from M, prob and energy for S1 group
        :return: none
        """
        # Compute energy
        self.prob, _, _, _, _, _ = self.fft.synthesize(self.M)
        self.prob = torch.where(self.prob.real > 0, self.prob.real, torch.tensor(1e-8, device=self.prob.device))
        self.energy = torch.log(self.prob)
        # Compute eta
        _, _, _, _, _, self.eta = self.fft.analyze(self.energy)

    # def compute_moments_lnz(self, eta: torch.Tensor, update: bool = True) -> Tuple[torch.Tensor, float]:
    #     """
    #     Compute the moments of the distribution
    #     :param eta: canonical parameters in log prob space of distribution
    #     :param update: whether to update the moments of the distribution and log partition constant
    #     :return: moments and log partition constant
    #     """
    #     negative_energy, _, _, _, _, _ = self.fft.synthesize(eta)
    #     maximum = torch.max(negative_energy.real)
    #     _, _, _, _, _, unnormalized_moments = self.fft.analyze(torch.exp(negative_energy - maximum))
    #     # TODO: Figure out why z_0 is the 0 index.
    #     z_0 = 0
    #     z_1 = unnormalized_moments.shape[1] // 2
    #     z_2 = unnormalized_moments.shape[2] // 2
    #     # Scale by invariant haar measure
    #     # Haar measure in Chirikjian's book for S1, semi-direct product R^2 \cross S1
    #     unnormalized_moments[z_0, z_1, z_2] *= torch.tensor(math.pi * 2, device=unnormalized_moments.device)
    #     moments = unnormalized_moments / unnormalized_moments[z_0, z_1, z_2]
    #     moments[z_0, z_1, z_2] = unnormalized_moments[z_0, z_1, z_2]
    #     l_n_z = torch.log(unnormalized_moments[z_0, z_1, z_2]) + maximum
    #     # Update moments of distribution and constant only when needed
    #     if update:
    #         self.moments = moments
    #         self.l_n_z = l_n_z.real
    #     return moments, l_n_z.real

    def compute_moments_lnz(self, eta: torch.Tensor, update: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: moments and log partition constant
        """
        energy, _, _, _, _, _ = self.fft.synthesize(eta)
        minimum = torch.min(torch.abs(energy)).to(energy.device)
        _, _, _, _, _, unnormalized_moments = self.fft.analyze(torch.exp(energy - minimum))
        z_0 = 0
        z_1 = unnormalized_moments.shape[2] // 2
        z_2 = unnormalized_moments.shape[3] // 2
        constant = (unnormalized_moments[:, z_0, z_1, z_2] * (math.pi * 2)).to(energy.device)
        moments = unnormalized_moments / constant.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        moments[:, z_0, z_1, z_2] = constant
        l_n_z = (torch.log(constant) + minimum).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if update:
            self.moments = moments
            self.l_n_z = l_n_z.real
        # Update moments of distribution and constant only when needed
        return moments, l_n_z.real

    def compute_energy(self, t: torch.Tensor) -> torch.Tensor:
        """
        Energy of the distribution
        :param t: samples
        :return: energy of the distribution
        """
        raise NotImplementedError("This distribution does not have closed form to compute energy from samples, "
                                  "instead use `normalize` to obtain energy from eta")

    def update_params(self) -> None:
        """
        Update parameters of the distribution
        :return: none
        """
        raise NotImplementedError("This class does not have natural parameters thus they cannot be updated")

    # # TODO: Get rid of this once SE2 real fft is implemented.
    # @property
    # def energy(self) -> torch.Tensor:
    #     if self._energy is None:
    #         self._energy, _, _, _, _, _ = self.fft.synthesize(self.eta)
    #     return self._energy.real

    # # TODO: Get rid of this once SE2 real fft is implemented.
    # @energy.setter
    # def energy(self, energy: torch.Tensor) -> None:
    #     self._energy = energy.clone()

    # # TODO: Get rid of this once SE2 real fft is implemented.
    # @property
    # def prob(self) -> torch.Tensor:
    #     if self._prob is None:
    #         self._prob, _, _, _, _, _ = self.fft.synthesize(self.M)
    #         self._prob = torch.where(self._prob.real > 0, self._prob.real, torch.tensor(1e-8, device=self._prob.device))
    #     return self._prob

    # # TODO: Get rid of this once SE2 real fft is implemented.
    # @prob.setter
    # def prob(self, prob: torch.Tensor) -> None:
    #     self._prob = prob.clone()


class SE2Gaussian(SE2):
    """ Class to represent Gaussian-Like distributions in SE2. """

    def __init__(self,
                 mu: torch.Tensor = torch.zeros(3),
                 cov: torch.Tensor = torch.eye(3),
                 inv_cov: torch.Tensor = torch.eye(3),
                 **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.cov = cov # [3, 3]
        # self.inv_cov = torch.inverse(cov)
        self.inv_cov = inv_cov.to(mu.device)
        self.from_samples()

    def update_params(self) -> None:
        raise NotImplementedError

    # def compute_energy(self, x):
    #     # assert x.shape[1] == 3
    #     diff = x - self.mu.unsqueeze(1)
    #     # Wrap angle
    #     diff[:, :,  2] = (diff[:, :,  2] + torch.pi) % (2 * torch.pi) - torch.pi
        
    #     # Using PyTorch's MultivariateNormal for log_prob calculation
    #     distribution = MultivariateNormal(loc=torch.zeros(3, device=x.device), 
    #                                     covariance_matrix=self.cov)
    #     logpdf = distribution.log_prob(diff)

    #     return logpdf.reshape(-1, *self.fft.spatial_grid_size)

    
    def compute_energy(self, x):
        k = x.shape[2]
        diff = x - self.mu.unsqueeze(1)
        # diff[:, :,  2] = (diff[:, :,  2] + torch.pi) % (2 * torch.pi) - torch.pi
        log_det = torch.logdet(self.cov) # [1]
        diff_expanded = diff.unsqueeze(-1) # Shape: [batch, 8000, 3, 1]
        mahalanobis_dist = torch.matmul(torch.matmul(diff_expanded.transpose(-1, -2), self.inv_cov), diff_expanded).squeeze(-1)
        # mahalanobis_dist = torch.sum(diff @ precision_matrix @ diff, dim=1).unsqueeze(-1)
        
        log_constant = k * torch.log(torch.tensor(2 * torch.pi)) + log_det 
        log_constant_ = torch.tile(log_constant.unsqueeze(0).unsqueeze(0), [x.shape[0], x.shape[1], 1]) # [batch, 8000, 1]
        log_likelihood = -0.5 * (log_constant_ + mahalanobis_dist)

        return log_likelihood.reshape(-1, *self.fft.spatial_grid_size)


class SE2MultimodalGaussian(SE2):
    """
    Class to represent a mixture of Gaussians for SE(2).
    """

    def __init__(self, mu_list: List[torch.Tensor] = [torch.zeros(3), torch.zeros(3)],
                 cov_list: List[torch.Tensor] = [torch.eye(3), torch.eye(3) * 0.1],
                 **kwargs):
        super().__init__(**kwargs)
        # List of distributions
        self.distributions = [SE2Gaussian(mu=mu, cov=cov, **kwargs) for mu, cov in zip(mu_list, cov_list)]
        # Compute number of components
        self.n_components = len(self.distributions)
        self.from_samples()

    def compute_energy(self, x):
        energy = torch.zeros(x.shape[0], device=x.device)
        log_weight = torch.log(torch.tensor(1.0 / self.n_components, device=x.device))
        for dist in self.distributions:
            energy += torch.exp(log_weight + dist.compute_energy(x))
        # Add small constant to avoid log(0)
        energy = torch.log(energy + 1e-9)
        return energy


class SE2Square(SE2):
    def __init__(self, x_limits: List[float], y_limits: List[float], theta_limits: List[float], scale: float, **kwargs):
        super().__init__(**kwargs)
        # Bounds for the square
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.theta_limits = theta_limits
        self.scale = scale
        self.from_samples()

    def compute_energy(self, x):
        """Square function in XY for SE2"""
        x_energy = torch.logical_and(torch.tensor(self.x_limits[0], device=x.device) < x[:, 0], 
                                     x[:, 0] < torch.tensor(self.x_limits[1], device=x.device))
        y_energy = torch.logical_and(torch.tensor(self.y_limits[0], device=x.device) < x[:, 1], 
                                     x[:, 1] < torch.tensor(self.y_limits[1], device=x.device))

        # Wrap angle from -pi to pi
        diff_t = (x[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
        
        t_energy = torch.logical_and(torch.tensor(self.theta_limits[0], device=x.device) < diff_t, 
                                     diff_t < torch.tensor(self.theta_limits[1], device=x.device))
        
        energy = torch.logical_and(torch.logical_and(x_energy, y_energy), t_energy) * self.scale
        return energy.reshape(self.fft.spatial_grid_size)