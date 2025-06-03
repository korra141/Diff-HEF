import torch
from copy import deepcopy
from typing import Type, List
from src.distributions.SE2.distribution_base import HarmonicExponentialDistribution
import pdb

import torch
from copy import deepcopy

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

class BayesFilter:
    def __init__(self, distribution: Type[HarmonicExponentialDistribution], prior: HarmonicExponentialDistribution, device):
        self.distribution = distribution
        self.prior = prior
        self.device = device

    def prediction(self, motion_model: HarmonicExponentialDistribution) -> HarmonicExponentialDistribution:
        # pdb.set_trace()
        predict = self.distribution.convolve(motion_model, self.prior)
        _, _ = predict.compute_moments_lnz(predict.eta, update=True)
        prob = torch.exp(predict.energy - predict.l_n_z) + 1e-8
        _, _, _, _, _, predict.M = predict.fft.analyze(prob)
        predict.prob, _, _, _, _, _ = predict.fft.synthesize(predict.M)
        predict.prob = torch.where(predict.prob.real > 0, predict.prob.real, torch.tensor(1e-8))
        # predict.energy = torch.log(predict.prob) 
        # _, _, _, _, _, predict.eta = predict.fft.analyze(predict.energy)
        self.prior = deepcopy(predict)
        return predict

    def update(self, measurement_model: HarmonicExponentialDistribution) -> HarmonicExponentialDistribution:
        update = self.distribution.product(self.prior, measurement_model)
        _, _ = update.compute_moments_lnz(update.eta.to(self.device), update=True)
        energy, _, _, _, _, _ = update.fft.synthesize(update.eta.to(self.device))
        prob = (torch.exp(energy.real - update.l_n_z) + 1e-8).to(self.device)
        _, _, _, _, _, update.M = update.fft.analyze(prob.to(self.device))
        update.energy = energy.real.to(self.device)
        update.prob, _, _, _, _, _ = update.fft.synthesize(update.M.to(self.device))
        update.prob = torch.where(update.prob.real > 0, update.prob.real.to(self.device), torch.tensor(1e-8, device=self.device))
        detached_update = detach_tensors(update)
        self.prior = deepcopy(detached_update)
        # self.prior = deepcopy(update)
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