import torch
import math
from src.distributions.SE2.GaussianDistribution import GaussianSE2
from src.distributions.SE2.SE2_FFT import SE2_FFT
import pdb

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

    def predict(self, prior, process):
        _,z_1 = self.fft.compute_moments_lnz(torch.log(prior + 1e-8))
        _,z_2 = self.fft.compute_moments_lnz(torch.log(process +  1e-8))
        prob_1 = prior/z_1
        prob_2 = process/z_2
        density_convolve = self.convolve(prob_1, prob_2)
        f, f1c, f1p, f2, f2f, eta = self.fft.analyze(torch.log(density_convolve + 1e-8))

        return eta, density_convolve

    def convolve(self, prob_1, prob_2):
        # padded_length = (2*self.band_limit[0] - 1,2*self.band_limit[1] - 1)
        # prob_1 = self.pad_for_fft_2d(prob_1, padded_length)
        # prob_2 = self.pad_for_fft_2d(prob_2, padded_length)
        f, f1c, f1p, f2, f2f, moments_1 = self.fft.analyze(prob_1)
        f, f1c, f1p, f2, f2f, moments_2 = self.fft.analyze(prob_2)
        moments_convolve = moments_1 * moments_2
        unnorm_density_convolve, f1c, f1p, f2, f2f, f_hat = self.fft.synthesize(moments_convolve)
        # unnorm_density_convolve_final = unnorm_density_convolve[:,math.floor(self.band_limit[0]/2):self.band_limit[0] +  math.floor(self.band_limit[0]/2),math.floor(self.band_limit[1]/2):self.band_limit[1] +  math.floor(self.band_limit[1]/2)].real
        # unnorm_density_convolve_final = unnorm_density_convolve[:,:self.band_limit[0],:self.band_limit[1]].real
        # unnorm_density_convolve = torch.clamp(unnorm_density_convolve,min=1e-8)
        # z_3 = self.normalisation_constant(unnorm_density_convolve)
        # z_3 = ((moments_convolve[:,self.band_limit[0]//2,self.band_limit[1]//2].real)/math.prod(self.band_limit)).unsqueeze(-1).unsqueeze(-1)
        _, z_3 = self.fft.compute_moments_lnz(torch.log(unnorm_density_convolve))
        density_convolve = abs(unnorm_density_convolve)/z_3

        return density_convolve

    def convert_from_energy_eta(self,energy_samples):
        # z = self.normalisation_constant(density_samples)
        # density_samples_norm = density_samples/z
        # energy_samples = torch.log(density_samples + 1e-10)
        # pdb.set_trace()
        f, f1c, f1p, f2, f2f, eta = self.fft.analyze(energy_samples)
        return eta

    def convert_from_eta_energy(self, eta):
        # pdb.set_trace()
        energy, f1c, f1p, f2, f2f, f_hat = self.fft.synthesize(eta)
        lnz , z_ = self.fft.compute_moments_lnz(energy)
        # density = torch.exp(energy)/z_
        energy_norm = energy - lnz
        return energy_norm

    def update(self,predict, measurements):
        eta = predict + self.convert_from_energy_eta(measurements)
        return self.convert_from_eta_energy(eta)


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