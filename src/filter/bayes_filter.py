import torch
from copy import deepcopy
from typing import Type, List
from src.distributions.SE2.distribution_base import HarmonicExponentialDistribution
import pdb

import torch
from copy import deepcopy
import psutil
import os
import gc

device = torch.device("cpu")

def detach_tensors(obj):
    """
    Recursively detach all tensors inside a Python object (dict, list, tuple, custom objects).
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_tensors(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_tensors(v) for v in obj)
    elif hasattr(obj, '__dict__'):  # If it's a custom object
        # Create a shallow copy first
        new_obj = obj.__class__.__new__(obj.__class__)
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, detach_tensors(v))
        return new_obj
    else:
        return obj
    
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

class BayesFilter:
    def __init__(self, distribution: Type[HarmonicExponentialDistribution], prior: HarmonicExponentialDistribution, device):
        self.distribution = distribution
        # self.prior = prior
        self.device = device

    def prediction(self, prior, motion_model: HarmonicExponentialDistribution) -> HarmonicExponentialDistribution:
        # pdb.set_trace()
        # print_memory_usage("Before Prediction")
        torch.cuda.empty_cache()
        gc.collect()
        # predict = self.distribution.convolve(motion_model, self.prior)
        predict = self.distribution.convolve_optimized(motion_model, prior, method='chunked')
        torch.cuda.empty_cache()
        gc.collect()
        # print_memory_usage("After Convolution")
        l_n_z = predict.compute_moments_lnz(predict.eta, update=True)
        # print_memory_usage("After Normalization Constant")
        prob = torch.exp(predict.energy - l_n_z) + 1e-8
        # print_memory_usage('line 62')
        del l_n_z
        torch.cuda.empty_cache()
        M_temp = predict.fft.analyze(prob)
        # print_memory_usage('line 64')
        predict.prob = predict.fft.synthesize(M_temp)
        # print_memory_usage('line 67')
        predict.prob = torch.where(predict.prob.real > 0, predict.prob.real, torch.tensor(1e-8))
        # print_memory_usage('line 68')
        predict.M = M_temp
        # predict.energy = torch.log(predict.prob) 
        # _, _, _, _, _, predict.eta = predict.fft.analyze(predict.energy)
        # self.prior = deepcopy(predict)
        # self.prior = predict
        # print_memory_usage('deep copy')
        return predict

    def update(self, prior, measurement_model: HarmonicExponentialDistribution) -> HarmonicExponentialDistribution:
        # print_memory_usage("Before Update")
        update = self.distribution.product(prior, measurement_model)
        # print_memory_usage("After Product")
        l_n_z = update.compute_moments_lnz(update.eta.to(self.device), update=True)
        # print_memory_usage("After Normalization Constant")
        energy= update.fft.synthesize(update.eta.to(self.device))
        # print_memory_usage("After Synthesize")
        prob = (torch.exp(energy.real - l_n_z) + 1e-8).to(self.device)
        # print_memory_usage("After Probability Calculation")
        M_temp  = update.fft.analyze(prob.to(self.device))
        # print_memory_usage("After Analyze")
        update.energy = energy.real.to(self.device)
        update.prob = update.fft.synthesize(M_temp.to(self.device))
        # print_memory_usage("After Final Synthesize")
        update.prob = torch.where(update.prob.real > 0, update.prob.real.to(self.device), torch.tensor(1e-8, device=self.device))
        update.M = M_temp.to(self.device)
        # detached_update = detach_tensors(update)
        # self.prior = deepcopy(detached_update)
        # self.prior = detached_update
        # print_memory_usage("updating prior")
        # # self.prior = deepcopy(update)
        return update

    @staticmethod
    def neg_log_likelihood(eta: torch.Tensor, l_n_z: float, pose: torch.Tensor, fft) -> float:
        dx, dy = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1)
        d_theta = pose[:, 2].unsqueeze(1)
        _, _, _, f_p_psi_m, _, _ = fft.synthesize(eta)
        f_p_psi_m = torch.fft.ifftshift(f_p_psi_m, dim=2)
        t_theta = 2 * torch.pi
        omega_n = 2 * torch.pi * (1 / t_theta) * torch.arange(f_p_psi_m.shape[2]).unsqueeze(0)
        f_p_psi = torch.sum(f_p_psi_m * torch.exp(-1j * omega_n * d_theta), dim=2)
        f_p_p = fft.resample_p2c_3d(f_p_psi)
        angle_x = 1j * 2 * torch.pi * (1 / 1.0) * torch.arange(f_p_p.shape[0]).unsqueeze(1) * dx
        angle_y = 1j * 2 * torch.pi * (1 / 1.0) * torch.arange(f_p_p.shape[1]).unsqueeze(1) * dy
        angle = angle_x + angle_y
        f = torch.sum(f_p_p * torch.exp(angle), dim=(0, 1)).real - l_n_z
        return -f.item()