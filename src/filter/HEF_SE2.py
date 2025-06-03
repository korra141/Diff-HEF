import torch
import math
from src.distributions.SE2.GaussianDistribution import GaussianSE2
from src.distributions.SE2.SE2_torch import SE2_FFT
import pdb
import time
import datetime

class HEFilter():
    def __init__(self, band_limit,range_x,range_y):
        super(HEFilter, self).__init__()
        self.band_limit = band_limit
        self.range_x = range_x
        self.range_y = range_y
        self.range_x_diff = self.range_x[1] - self.range_x[0]
        self.range_y_diff = self.range_y[1] - self.range_y[0]
        self.fft = SE2_FFT(spatial_grid_size=self.band_limit,
                  interpolation_method='spline',
                  spline_order=1,
                  oversampling_factor=1)

    # def pad_for_fft_2d(self, tensor, target_shape):
    #     pad_h = target_shape[0] - tensor.shape[1]
    #     pad_w = target_shape[1] - tensor.shape[2]
    #     # Padding format in PyTorch: (left, right, top, bottom)
    #     # padded_tensor = torch.nn.functional.pad(tensor, (math.ceil(pad_w/2), pad_w - math.ceil(pad_w/2),math.ceil(pad_h/2), pad_h - math.ceil(pad_h/2)), mode='constant', value=0)
    #     padded_tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    #     return padded_tensor

    @staticmethod
    def mul(fh1, fh2):
        assert fh1.shape == fh2.shape

        p0 = fh1.shape[1] // 2  # Indices of the zero frequency component
        q0 = fh1.shape[2] // 2

        a = p0 - q0
        b = int(p0 + torch.ceil(torch.tensor(fh2.shape[2] / 2.)))

        fh12 = torch.einsum('rpn,rnn->rpn', fh1, fh2[:, a:b, :])
        
        return fh12

    @staticmethod
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
                batch_result.append(prod[:, a:b])
            batch_result = torch.stack(batch_result, dim=0)
            results.append(batch_result)
        fh12 = torch.stack(results, dim=0)

        return fh12

    def predict(self, prior, process):
        _,z_1 = self.fft.compute_moments_lnz(torch.log(prior + 1e-8))
        _,z_2 = self.fft.compute_moments_lnz(torch.log(process +  1e-8))
        prob_1 = prior.div_(z_1.view(-1, 1, 1, 1))
        prob_2 = process.div_(z_2.view(-1, 1, 1, 1))
        density_convolve = self.convolve(prob_1, prob_2)
        _, _, _, _, _, eta = self.fft.analyze(torch.log(density_convolve + 1e-8))
        
        return eta, density_convolve

    def convolve(self, prob_1, prob_2):
        # padded_length = (2*self.band_limit[0] - 1,2*self.band_limit[1] - 1)
        # prob_1 = self.pad_for_fft_2d(prob_1, padded_length)
        # prob_2 = self.pad_for_fft_2d(prob_2, padded_length)
        _, _, _, _, _, moments_1 = self.fft.analyze(prob_1)
        _, _, _, _, _, moments_2 = self.fft.analyze(prob_2)
        # moments_convolve = moments_1 * moments_2
        moments_convolve = self.mulT(moments_1, moments_2)
        start_time = datetime.datetime.now()
        unnorm_density_convolve, _, _, _, _, _ = self.fft.synthesize(moments_convolve)
        # print("Time for synthesize: ", datetime.datetime.now() - start_time)
        # unnorm_density_convolve_final = unnorm_density_convolve[:,math.floor(self.band_limit[0]/2):self.band_limit[0] +  math.floor(self.band_limit[0]/2),math.floor(self.band_limit[1]/2):self.band_limit[1] +  math.floor(self.band_limit[1]/2)].real
        # unnorm_density_convolve_final = unnorm_density_convolve[:,:self.band_limit[0],:self.band_limit[1]].real
        # unnorm_density_convolve = torch.clamp(unnorm_density_convolve,min=1e-8)
        # z_3 = self.normalisation_constant(unnorm_density_convolve)
        # z_3 = ((moments_convolve[:,self.band_limit[0]//2,self.band_limit[1]//2].real)/math.prod(self.band_limit)).unsqueeze(-1).unsqueeze(-1)
        _, z_3 = self.fft.compute_moments_lnz(torch.log(unnorm_density_convolve))
        density_convolve = abs(unnorm_density_convolve).div_(z_3.view(-1, 1, 1, 1))
        # print("Time for compute_moments_lnz and normalization: ", datetime.datetime.now() - start_time_nc)

        return density_convolve

    def convert_from_energy_eta(self,energy_samples):
        # z = self.normalisation_constant(density_samples)
        # density_samples_norm = density_samples/z
        # energy_samples = torch.log(density_samples + 1e-10)
        # pdb.set_trace()
        _, _, _, _, _, eta = self.fft.analyze(energy_samples)
        return eta

    def convert_from_eta_energy(self, eta):
        # pdb.set_trace()
        energy, _, _, _, _, _ = self.fft.synthesize(eta)
        lnz , z_ = self.fft.compute_moments_lnz(energy)
        # density = torch.exp(energy)/z_
        energy_norm = energy.real - lnz.view(-1, 1, 1, 1)
        return energy_norm

    def update(self,predict, measurements):
        start_time = datetime.datetime.now()
        eta = predict + self.convert_from_energy_eta(measurements)
        result = self.convert_from_eta_energy(eta)
        # print("Time for update step: ", datetime.datetime.now() - start_time)
        return result

