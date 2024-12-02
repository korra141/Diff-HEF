import torch
import math
import numpy as np
# from scipy.stats import norm
import pdb

class HarmonicExponentialDistribution:
    def __init__(self, range_x, range_y, bandwidth, step_size):
        self.range_x = range_x
        self.range_y = range_y
        self.grid_size = bandwidth
        self.step_t = step_size
        self.range_x_diff = self.range_x[1] - self.range_x[0]
        self.range_y_diff = self.range_y[1] - self.range_y[0]

    def negative_log_likelihood_density(self,density,value):
        """
        Computes the negative log-likelihood density for a given density and value.
        Args:
            density (torch.Tensor): The density tensor of shape [batchsize, grid_size1, grid_size2].
            value (torch.Tensor): The value tensor of shape [batchsize, 2], where each row represents a point (x, y).
        Returns:
            torch.Tensor: The mean negative log-likelihood density.
        Note:
            The value should lie on the grid. If it does not, this function will output an incorrect value since it does not use an interpolator.
        """
        
        id_x = ((value[:,0] - self.range_x[0])/self.step_t[0]).to(torch.int).to(torch.float) # [batchsize]
        id_y = ((value[:,1] - self.range_y[0])/self.step_t[1]).to(torch.int).to(torch.float)
        

        # It is more numerically stable to compute the log of the density and then exponentiate it.

        energy = torch.log(density + 1e-40)
        z = self.normalization_constant(density)
        energy = energy  + torch.log(z)
      
        eta = torch.fft.fft2(energy)

        k_x = torch.arange(self.grid_size[0]) 
        k_y = torch.arange(self.grid_size[1])

        temp_x = id_x.unsqueeze(-1) * k_x.unsqueeze(0) 
        temp_y = id_y.unsqueeze(-1) * k_y.unsqueeze(0) 

        exp_x = torch.exp(2j * math.pi * temp_x/self.grid_size[0])
        exp_y = torch.exp(2j * math.pi * temp_y/self.grid_size[1])

        temp_eta_y = torch.bmm(eta,exp_y.unsqueeze(-1)).squeeze(-1)

        reconstructed = (torch.bmm(temp_eta_y.unsqueeze(1), exp_x.unsqueeze(-1)) / math.prod(self.grid_size)).real

        # z = (torch.sum(density,dim=(1,2))*((range_x_diff * range_y_diff)/math.prod(grid_size))).unsqueeze(-1).unsqueeze(-1)

        result = -reconstructed + torch.log(z)

        # print("Integrating the distribution",torch.mean(torch.sum(density/z,dim=(1,2))*step_t[0]*step_t[1]))

        return torch.mean(result.squeeze(-1))
    
    def loss_energy(self,density,value):
        """
        Computes the negative log-likelihood density for a given density and value.
        Args:
            density (torch.Tensor): The density tensor of shape [batchsize, grid_size1, grid_size2].
            value (torch.Tensor): The value tensor of shape [batchsize, 2], where each row represents a point (x, y).
        Returns:
            torch.Tensor: The mean negative log-likelihood density.
        Note:
            The value should lie on the grid. If it does not, this function will output an incorrect value since it does not use an interpolator.
        """
        
        # id_x = ((value[:,0] - self.range_x[0])/self.step_t[0]).to(torch.int).to(torch.float) # [batchsize]
        # id_y = ((value[:,1] - self.range_y[0])/self.step_t[1]).to(torch.int).to(torch.float)
        

        # It is more numerically stable to compute the log of the density and then exponentiate it.

        energy = torch.log(density + 1e-40)
        ln_z = self.normalization_constant_energy(energy)
        # print(z)
        # energy = energy  + torch.log(z)
      
        eta = torch.fft.fft2(energy)

        k_x = torch.arange(self.grid_size[0]) 
        k_y = torch.arange(self.grid_size[1])

        x = (value[:,0] - self.range_x[0])/self.range_x_diff
        y = (value[:,1] - self.range_y[0])/self.range_y_diff

        temp_x = x.unsqueeze(-1) * k_x.unsqueeze(0) 
        temp_y = y.unsqueeze(-1) * k_y.unsqueeze(0) 

        exp_x = torch.exp(2j * math.pi * temp_x)
        exp_y = torch.exp(2j * math.pi * temp_y)

        temp_eta_y = torch.bmm(eta,exp_y.unsqueeze(-1)).squeeze(-1)

        reconstructed = (torch.bmm(temp_eta_y.unsqueeze(1), exp_x.unsqueeze(-1))/math.prod(self.grid_size)).real

        # z = (torch.sum(density,dim=(1,2))*((range_x_diff * range_y_diff)/math.prod(grid_size))).unsqueeze(-1).unsqueeze(-1)

        result = -reconstructed + ln_z

        # print("Integrating the distribution",torch.mean(torch.sum(density/z,dim=(1,2))*step_t[0]*step_t[1]))

        return torch.mean(result.squeeze(-1))

    def normalization_constant(self,density):
        moments = torch.fft.fft2(density)
        z = (moments[:,0,0] * ((self.range_x_diff * self.range_y_diff) / math.prod(self.grid_size))).unsqueeze(-1).unsqueeze(-1)
        return z.real
    
    def normalization_constant_energy(self,energy):
        # min_energy = torch.min(energy, dim=-1, keepdim=True).values
        # min_min_energy = torch.min(min_energy, dim=1, keepdim=True).values
        # print(min_min_energy)
        moments = torch.fft.fft2(torch.exp(energy))
        # print(moments.isnan().any())
        z = torch.log((moments[:,0,0].real * ((self.range_x_diff * self.range_y_diff) / math.prod(self.grid_size))).unsqueeze(-1).unsqueeze(-1))
        return z
    
    def pad_for_fft_2d(self,tensor, target_shape):
        pad_h = target_shape[0] - tensor.shape[1]
        pad_w = target_shape[1] - tensor.shape[2]
        # Padding format in PyTorch: (left, right, top, bottom)
        # padded_tensor = torch.nn.functional.pad(tensor, (math.ceil(pad_w/2), pad_w - math.ceil(pad_w/2),math.ceil(pad_h/2), pad_h - math.ceil(pad_h/2)), mode='constant', value=0)
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return padded_tensor
    
    def convolve(self,prob_1, prob_2):
            padded_length = (2*self.grid_size[0] - 1,2*self.grid_size[1] - 1)
            prob_1 = self.pad_for_fft_2d(prob_1, padded_length)
            prob_2 = self.pad_for_fft_2d(prob_2, padded_length)
            moments_1 = torch.fft.fft2(prob_1)
            moments_2 = torch.fft.fft2(prob_2)
            moments_convolve = moments_1 * moments_2
            unnorm_density_convolve = torch.fft.ifft2(moments_convolve)
            unnorm_density_convolve_final = unnorm_density_convolve[:,math.floor(self.grid_size[0]/2):math.floor(self.grid_size[0]/2) + self.grid_size[0] ,math.floor(self.grid_size[1]/2):math.floor(self.grid_size[1]/2) + self.grid_size[1]].real
            unnorm_density_convolve_final = torch.clamp(unnorm_density_convolve_final,min=1e-10)
            z_3 = self.normalization_constant(unnorm_density_convolve_final)
            density_convolve = unnorm_density_convolve_final/z_3
            return density_convolve

    def mode(self,predicted_density):

        max_vals_row, max_x = torch.max(predicted_density, dim=2)  # max over columns, shape (10, 50)

        # Step 2: Get the maximum values and their column indices along the second-to-last dimension (dim=1)
        _, max_y = torch.max(max_vals_row, dim=1)  # max over rows, shape (10,)

        # Step 3: Gather the x indices corresponding to the max y indices
        max_x = max_x[torch.arange(max_x.size(0)), max_y]  # shape (10,)

        # Step 4: Stack the coordinates to get the final shape (10, 2)
        mode_idx = torch.stack((max_x, max_y), dim=1) 
    
        poses_mode_x = mode_idx[:, 0] * self.step_t[0] + self.range_x[0]
        poses_mode_y = mode_idx[:, 1] * self.step_t[1] + self.range_y[0]
        poses_mode = torch.stack((poses_mode_x, poses_mode_y), dim=-1)

        return poses_mode


    # def product(self, other_distribution):
    #     return self.pdf() * other_distribution.pdf()

    # def pdf(self):
    #     return norm.pdf(self.x_grid, loc=self.mean, scale=self.bandwidth) * \
    #            norm.pdf(self.y_grid, loc=self.mean, scale=self.bandwidth)

