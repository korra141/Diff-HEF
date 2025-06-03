import torch
from abc import ABC, abstractmethod
from typing import Type, Tuple

class HarmonicExponentialDistribution(ABC):
    def __init__(self, samples: torch.Tensor, fft: Type):
        self._eta = None
        self._M = None
        self._energy = None
        self._prob = None
        self.samples = samples
        self.moments = None
        self.l_n_z = None
        self.fft = fft

    def from_samples(self) -> None:
        energy = self.compute_energy(self.samples)
        _, _, _, _, _,self.eta = self.fft.analyze(energy)
        self.energy , _, _, _, _, _ = self.fft.synthesize(self.eta)

    @classmethod
    def from_eta(cls, eta: torch.Tensor, fft: Type) -> Type['HarmonicExponentialDistribution']:
        dist = cls(samples=None, fft=fft)
        dist.eta = eta
        return dist

    @classmethod
    def from_M(cls, M: torch.Tensor, fft: Type) -> Type['HarmonicExponentialDistribution']:
        dist = cls(samples=None, fft=fft)
        dist.M = M
        dist.compute_eta()
        return dist

    # @abstractmethod
    # def normalize(self):
    #     pass

    @abstractmethod
    def compute_energy(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_moments_lnz(self, eta: torch.Tensor, update: bool = True) -> Tuple[torch.Tensor, float]:
        pass

    # @property
    # def prob(self) -> torch.Tensor:
    #     if self._prob is None:
    #         self._prob, _, _, _, _,_  = self.fft.synthesize(self.M).real
    #         self._prob = torch.where(self._prob > 0, self._prob, torch.tensor(1e-8))
    #     return self._prob

    # @prob.setter
    # def prob(self, prob: torch.Tensor) -> None:
    #     self._prob = prob.clone()

    # @property
    # def M(self) -> torch.Tensor:
    #     return self._M

    # @M.setter
    # def M(self, M: torch.Tensor) -> None:
    #     self._M = M.clone()
    #     if self._eta is None:
    #         self.compute_eta()

    # @property
    # def eta(self) -> torch.Tensor:
    #     # if self._eta is None:
    #     #     _, _, _, _, _, self._eta = self.fft.analyze(self.energy)
    #     return self._eta

    # @eta.setter
    # def eta(self, eta: torch.Tensor) -> None:
    #     self._eta = eta.clone()

    # @property
    # def energy(self) -> torch.Tensor:
    #     return self._energy

    # @energy.setter
    # def energy(self, energy: torch.Tensor) -> None:
    #     self._energy = energy.clone()