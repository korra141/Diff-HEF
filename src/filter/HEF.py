import torch
import math
from src.distributions.R2.StandardDistribution import GaussianDistribution

class HEFilter():
    def __init__(self, band_limit,range_x,range_y):
        super(HEFilter, self).__init__()
        self.band_limit = band_limit
        self.range_x = range_x
        self.range_y = range_y
        self.range_x_diff = self.range_x[1] - self.range_x[0]
        self.range_y_diff = self.range_y[1] - self.range_y[0]

    def pad_for_fft_2d(self, tensor, target_shape):
        pad_h = target_shape[0] - tensor.shape[1]
        pad_w = target_shape[1] - tensor.shape[2]
        # Padding format in PyTorch: (left, right, top, bottom)
        # padded_tensor = torch.nn.functional.pad(tensor, (math.ceil(pad_w/2), pad_w - math.ceil(pad_w/2),math.ceil(pad_h/2), pad_h - math.ceil(pad_h/2)), mode='constant', value=0)
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return padded_tensor

    def predict(self, prior, process):
        z_1 = self.normalisation_constant(prior)
        z_2 = self.normalisation_constant(process)
        prob_1 = prior/z_1
        prob_2 = process/z_2
        density_convolve = self.convolve(prob_1, prob_2)
        eta = torch.fft.fft2(torch.log(density_convolve + 1e-8))

        return eta, density_convolve

    def convolve(self, prob_1, prob_2):
        # padded_length = (2*self.band_limit[0] - 1,2*self.band_limit[1] - 1)
        # prob_1 = self.pad_for_fft_2d(prob_1, padded_length)
        # prob_2 = self.pad_for_fft_2d(prob_2, padded_length)
        moments_1 = torch.fft.fftshift(torch.fft.fft2(prob_1),dim=(1,2))
        moments_2 = torch.fft.fftshift(torch.fft.fft2(prob_2),dim=(1,2))
        moments_convolve = moments_1 * moments_2
        unnorm_density_convolve = torch.fft.ifftshift(torch.fft.ifft2(moments_convolve),dim=(1,2)).real
        # unnorm_density_convolve_final = unnorm_density_convolve[:,math.floor(self.band_limit[0]/2):self.band_limit[0] +  math.floor(self.band_limit[0]/2),math.floor(self.band_limit[1]/2):self.band_limit[1] +  math.floor(self.band_limit[1]/2)].real
        # unnorm_density_convolve_final = unnorm_density_convolve[:,:self.band_limit[0],:self.band_limit[1]].real
        # unnorm_density_convolve = torch.clamp(unnorm_density_convolve,min=1e-8)
        # z_3 = self.normalisation_constant(unnorm_density_convolve)
        z_3 = ((moments_convolve[:,self.band_limit[0]//2,self.band_limit[1]//2].real)/math.prod(self.band_limit)).unsqueeze(-1).unsqueeze(-1)
        density_convolve = abs(unnorm_density_convolve)/z_3

        return density_convolve

    def convert_from_energy_eta(self,density_samples):
        # z = self.normalisation_constant(density_samples)
        # density_samples_norm = density_samples/z
        energy_samples = torch.log(density_samples + 1e-10)
        eta = torch.fft.fft2(energy_samples)
        return eta

    def convert_from_eta_energy(self, eta):
        energy = torch.fft.ifft2(eta)
        prob_unnorm = torch.exp(energy.real)
        z_ = self.normalisation_constant(prob_unnorm)
        density = prob_unnorm/z_
        return density.real

    def update(self,predict, measurements):
        eta = predict + self.convert_from_energy_eta(measurements)
        return self.convert_from_eta_energy(eta)

    def normalisation_constant(self,density):
        # maximum = torch.max(energy, dim=-1).values.unsqueeze(-1)
        moments = torch.fft.fftshift(torch.fft.fft2(density),dim=(1,2))
        z_ = torch.real((moments[:,self.band_limit[0]//2,self.band_limit[1]//2]*(self.range_x_diff*self.range_y_diff))/math.prod(self.band_limit)).unsqueeze(-1).unsqueeze(-1)
        return z_

    def analytic_filter(self,control,motion_noise,measurements,measurement_noise,prior_pdf):
        process = GaussianDistribution(control, motion_noise, self.range_x, self.range_y,self.band_limit)
        process_pdf  = process.density_over_grid()
        eta_bel_x_t_bar, density_bel_x_t_bar = self.predict(prior_pdf, process_pdf)
        density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32)
        measurement_likelihood = GaussianDistribution(measurements,measurement_noise,self.range_x,self.range_y,self.band_limit)
        measurement_pdf = measurement_likelihood.density_over_grid()    
        posterior_pdf = self.update(eta_bel_x_t_bar, measurement_pdf)
        posterior_pdf = posterior_pdf.to(torch.float32)
        return posterior_pdf, measurement_pdf,density_bel_x_t_bar