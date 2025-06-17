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
import psutil
import os

device = torch.device("cpu")

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    memory_in_mb = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2

        print(f"[{tag}] CUDA Memory | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak Allocated: {max_allocated:.2f} MB | Peak Reserved: {max_reserved:.2f} MB")

    print(f"[{tag}] Memory Usage: {memory_in_mb:.2f} MB")
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
        self.eta = self.fft.analyze(energy)
        # This seems redundant, but as there is a loss of information in FFT analyze due to cartesian to polar
        # interpolation, the normalization constant is computed wrt to the energy synthesize by the eta params and not
        # by the one originally used as input. Therefore, normalizing the "original" energy starts giving bad results
        self.energy = self.fft.synthesize(self.eta)

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
    

    def mulT_optimized_v1(fh1, fh2):
        """
        Vectorized version - eliminates nested loops
        """
        assert fh1.shape == fh2.shape
        
        batch_size, r, p, q = fh1.shape
        p0 = p // 2
        q0 = q // 2
        a = p0 - q0
        b = int(p0 + int(torch.ceil(torch.tensor(q / 2.))))
        
        # Vectorized batch matrix multiplication
        # fh1: [batch, r, p, q], fh2: [batch, r, q, p] (transposed)
        fh2_transposed = fh2.transpose(-2, -1)  # [batch, r, q, p]
        
        # Batch matrix multiplication: [batch, r, p, q] @ [batch, r, q, p] -> [batch, r, p, p]
        prod = torch.matmul(fh1, fh2_transposed)
        
        # Slice the result
        result = prod[:, :, :, a:b]
        
        return result

    def mulT_optimized_v2(fh1, fh2):
        """
        Memory-efficient version using einsum
        """
        assert fh1.shape == fh2.shape
        
        batch_size, r, p, q = fh1.shape
        p0 = p // 2
        q0 = q // 2
        a = p0 - q0
        b = int(p0 + int(torch.ceil(torch.tensor(q / 2.))))
        
        # Use einsum for efficient computation
        # 'brpq,brqk->brpk' where k is the sliced dimension
        fh2_sliced = fh2[:, :, a:b, :]  # Pre-slice fh2
        result = torch.einsum('brpq,brkq->brpk', fh1, fh2_sliced)
        
        return result

    def mulT_chunked(fh1, fh2, chunk_size=8):
        """
        Process in chunks to reduce peak memory usage
        """
        assert fh1.shape == fh2.shape
        
        batch_size, r, p, q = fh1.shape
        p0 = p // 2
        q0 = q // 2
        a = p0 - q0
        b = int(p0 + int(torch.ceil(torch.tensor(q / 2.))))
        
        results = []
        
        # Process batches in chunks
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Get chunk
            fh1_chunk = fh1[i:end_idx]
            fh2_chunk = fh2[i:end_idx]
            
            # Vectorized computation for chunk
            fh2_transposed = fh2_chunk.transpose(-2, -1)
            prod = torch.matmul(fh1_chunk, fh2_transposed)
            chunk_result = prod[:, :, :, a:b]
            
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)

    def mulT_inplace(fh1, fh2):
        """
        In-place computation to minimize memory allocation
        """
        assert fh1.shape == fh2.shape
        
        batch_size, r, p, q = fh1.shape
        p0 = p // 2
        q0 = q // 2
        a = p0 - q0
        b = int(p0 + int(torch.ceil(torch.tensor(q / 2.))))
        
        # Pre-allocate output tensor
        output_shape = (batch_size, r, p, b - a)
        fh12 = torch.empty(output_shape, dtype=fh1.dtype, device=fh1.device)
        
        # Process each batch to minimize memory usage
        for batch_idx in range(batch_size):
            # Compute matrix multiplication for entire batch at once
            prod = torch.matmul(fh1[batch_idx], fh2[batch_idx].transpose(-2, -1))
            fh12[batch_idx] = prod[:, a:b]
        
        return fh12

    @classmethod
    def convolve_optimized(cls, dist1, dist2, method='vectorized', chunk_size=10):
        """
        Optimized convolution with multiple memory-saving strategies
        """
        if method == 'vectorized':
            M = cls.mulT_optimized_v1(dist2.M, dist1.M)
        elif method == 'einsum':
            M = cls.mulT_optimized_v2(dist2.M, dist1.M)
        elif method == 'chunked':
            M = cls.mulT_chunked(dist2.M, dist1.M, chunk_size)
        elif method == 'inplace':
            M = cls.mulT_inplace(dist2.M, dist1.M)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return cls.from_M(M, dist1.fft)

    def normalize(self) -> None:
        """
        Updated moments, energy, probability, log partition constant and compute
        Ms for SE2 group
        :return: none
        """
        # print_memory_usage("Before Normalization")
        l_n_z = self.compute_moments_lnz(self.eta, update=True)
        # print_memory_usage("After Normalization Constant")
        prob = torch.exp(self.energy - l_n_z) + 1e-8
        # print_memory_usage("After Probability Computation")
        M_tmp = self.fft.analyze(prob)
        # print_memory_usage("After Ms Computation")
        prob = self.fft.synthesize(M_tmp)
        # print_memory_usage("After Ms Synthesis")
        self.prob = torch.where(prob.real > 0, prob.real, 1e-8)
        self.M = M_tmp
        # print_memory_usage("After Probability Update")

    def compute_eta(self) -> None:
        """
        Compute eta from M, prob and energy for S1 group
        :return: none
        """
        # Compute energy
        self.prob = self.fft.synthesize(self.M)
        self.prob = torch.where(self.prob.real > 0, self.prob.real, torch.tensor(1e-8, device=self.prob.device))
        self.energy = torch.log(self.prob)
        # Compute eta
        self.eta = self.fft.analyze(self.energy)

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
        energy = self.fft.synthesize(eta)
        minimum = torch.min(torch.abs(energy)).to(energy.device)
        unnormalized_moments = self.fft.analyze(torch.exp(energy - minimum))
        z_0 = 0
        z_1 = unnormalized_moments.shape[2] // 2
        z_2 = unnormalized_moments.shape[3] // 2
        constant = (unnormalized_moments[:, z_0, z_1, z_2] * (math.pi * 2)).to(energy.device)
        # moments = unnormalized_moments / constant.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # moments[:, z_0, z_1, z_2] = constant
        l_n_z = (torch.log(constant) + minimum).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Update moments of distribution and constant only when needed
        return l_n_z.real

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