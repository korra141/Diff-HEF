# Write the updated FFT code for SE2

import torch
import torch.fft as fft
import torch.nn.functional as F
from src.distributions.SE2.T1FFT import T1FFT
from src.distributions.SE2.T2FFT import T2FFT
from src.utils.interpolation import bilinear_interpolate_torch_with_nan, bilinear_interpolate_torch_circular_padded
from einops import rearrange
import math

import pdb
import torch
import torch.fft as fft
import torch.nn.functional as F
from src.distributions.SE2.T1FFT import T1FFT
from src.distributions.SE2.T2FFT import T2FFT
from einops import rearrange
import math

import pdb

def preprocess_input_with_cval(input, cval=float('nan')):
    """
    Extend the input tensor with a border filled with a constant value.
    
    Parameters:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        cval (float): The constant value to use for the extended border.
    
    Returns:
        torch.Tensor: Extended input tensor with out-of-bounds area filled with cval.
    """
    N, C, H, W = input.shape
    
    # Create an extended tensor with NaN-filled border
    extended = torch.full((N, C, H + 2, W + 2), cval, dtype=input.dtype, device=input.device)
    extended[:, :, 1:-1, 1:-1] = input  # Copy the original input into the center
    
    return extended

def adjust_grid_for_extension(grid):
    """
    Adjust the grid to account for the extended input tensor.
    
    Parameters:
        grid (torch.Tensor): Sampling grid of shape (N, H_out, W_out, 2).
    
    Returns:
        torch.Tensor: Adjusted grid of the same shape.
    """
    # Adjust grid to account for the one-pixel extension on all sides
    
    N, H, W, C = grid.shape
    adjusted_grid = grid.clone()
#     adjusted_grid[..., 0] = adjusted_grid[..., 0] * (H / (H + 10))
#     adjusted_grid[..., 1] = adjusted_grid[..., 1] * (W / (W + 10))
#     N, H, W, C = adjusted_grid.shape
    adjusted_grid[..., 0] = 2.0 * adjusted_grid[..., 0] / (W - 1) - 1.0  # Normalize x
    adjusted_grid[..., 1] = 2.0 * adjusted_grid[..., 1] / (H - 1) - 1.0  # Normalize y
    
    return adjusted_grid

def map_coordinates_torch_circular(input_tensor, coords):
    """
    Mimic scipy.ndimage.map_coordinates using PyTorch.
    :param input_tensor: Input tensor of shape (H, W).
    :param coords: Coordinates for sampling, shape (2, N) where N is the number of points.
    :param order: Interpolation order (only supports 1 for bilinear interpolation).
    :param mode: Boundary mode ('constant', 'nearest', 'reflect', 'wrap').
    :param cval: Constant value for 'constant' mode.
    :return: Interpolated values at the given coordinates.
    """

    fa = torch.empty((input_tensor.shape[0], input_tensor.shape[1] + 1, input_tensor.shape[2] + 1))
    fa[:, :-1, :-1] = input_tensor
    fa[:, -1, :-1] = input_tensor[:, 0, :]
    fa[:, :-1, -1] = input_tensor[:, :, 0]
    fa[:, -1, -1] = input_tensor[:, 0, 0]
    
#     fa = fa.unsqueeze(0)

    wrapped_coords_x = coords[0, ...] % input_tensor.shape[1]
    wrapped_coords_y = coords[1, ...] % input_tensor.shape[2]
    
#     hp = abs(fa.shape[1] - wrapped_coords_x.shape[1])
#     vp = abs(fa.shape[2] - wrapped_coords_x.shape[2])
    
    
                    
    
#     f = torch.nn.functional.pad(fa, (1, 1, 1, 1), mode='circular')
        
    sampled = bilinear_interpolate_torch_circular_padded(fa, wrapped_coords_x ,wrapped_coords_y)
    return sampled


def map_coordinates_torch_constant(input_tensor, coords):
    """
    Mimic scipy.ndimage.map_coordinates using PyTorch.
    :param input_tensor: Input tensor of shape (H, W).
    :param coords: Coordinates for sampling, shape (2, N) where N is the number of points.
    :param order: Interpolation order (only supports 1 for bilinear interpolation).
    :param mode: Boundary mode ('constant', 'nearest', 'reflect', 'wrap').
    :param cval: Constant value for 'constant' mode.
    :return: Interpolated values at the given coordinates.
    """
#     input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
#     fa = preprocess_input_with_cval(input_tensor, cval=cval)

    fa = torch.empty((input_tensor.shape[0],input_tensor.shape[1] + 1, input_tensor.shape[2] + 1))
    fa[:, :-1, :-1] = input_tensor
    fa[:, -1, :-1] = input_tensor[:, 0, :]
    fa[:, :-1, -1] = input_tensor[:, :, 0]
    fa[:, -1, -1] = input_tensor[:, 0, 0]
    
    wrapped_coords_x = coords[0, ...] % input_tensor.shape[1]
    wrapped_coords_y = coords[1, ...] % input_tensor.shape[2]
    
    f = torch.nn.functional.pad(fa,(1, 1, 1, 1),mode='constant', value=float('nan'))

    sampled = bilinear_interpolate_torch_with_nan(f, wrapped_coords_x ,wrapped_coords_y)
    return sampled

def shift_fft(f):
    nx = f.shape[1]
    ny = f.shape[2]
    p0, q0 = nx // 2, ny // 2

    X, Y = torch.meshgrid(
        torch.arange(p0, p0 + nx) % nx,
        torch.arange(q0, q0 + ny) % ny,
        indexing="ij",
    )

    fs = f[:,X, Y,...]
    return T2FFT.analyze(fs, axes=(1, 2))

def shift_ifft(fh):
    nx = fh.shape[1]
    ny = fh.shape[2]
    p0, q0 = nx // 2, ny // 2

    X, Y = torch.meshgrid(
        torch.arange(-p0, -p0 + nx,dtype=int) % nx,
        torch.arange(-q0, -q0 + ny,dtype=int) % ny,
        indexing="ij",
    )

    fs = T2FFT.synthesize(fh, axes=(1, 2))
    f = fs[:, X, Y,...]
    return f


class SE2_FFT():
    def __init__(
        self,
        spatial_grid_size,
        interpolation_method="spline",
        spline_order=1,
        oversampling_factor=1,
    ):
        self.spatial_grid_size = spatial_grid_size  # tau_x, tau_y, theta
        self.interpolation_method = interpolation_method

        if interpolation_method == "spline":
            self.spline_order = spline_order

            self.p0 = spatial_grid_size[0] // 2
            self.q0 = spatial_grid_size[1] // 2
            self.r_max = torch.sqrt(torch.tensor(self.p0**2 + self.q0**2))

            self.n_samples_r = int(oversampling_factor * (torch.ceil(self.r_max) + 1))
            self.n_samples_t = int(
                oversampling_factor * (torch.ceil(2 * torch.pi * self.r_max))
            )

            r = torch.linspace(0.0, self.r_max, self.n_samples_r)
            theta = torch.linspace(-torch.pi, torch.pi, self.n_samples_t + 1)[:-1]
            R, THETA = torch.meshgrid(r, theta, indexing="ij")

            X = R * torch.cos(THETA)
            Y = R * torch.sin(THETA)

            I = X + self.p0
            J = Y  + self.q0
            self.c2p_coords = torch.stack((I, J), dim=0)

            i = torch.arange(0, self.spatial_grid_size[0])
            j = torch.arange(0, self.spatial_grid_size[1])
            x = i  - self.p0
            y = j - self.q0
            X, Y = torch.meshgrid(x, y, indexing="ij")

            R = torch.sqrt(X**2 + Y**2)
            T = torch.atan2(Y, X)

#             R *= (self.n_samples_r - 1) / self.r_max
#             T *= self.n_samples_t / (2 * torch.pi)

            self.p2c_coords = torch.stack((R, T), dim=0)

        elif interpolation_method == "Fourier":
            r_max = 1.0 / torch.sqrt(torch.tensor(2.0))
            nr = 15 * torch.ceil(r_max * spatial_grid_size[0])
            nt = 5 * torch.ceil(2 * torch.pi * r_max * spatial_grid_size[0])
            nx, ny = spatial_grid_size[:2]
            self.flerp = FourierInterpolator.init_cartesian_to_polar(nr, nt, nx, ny)
        else:
            raise ValueError("Unknown interpolation method:" + str(interpolation_method))

    def analyze(self, f):
        f1c = shift_fft(f)
        f1p = self.resample_c2p_3d(f1c)

        f2 = T1FFT.analyze(f1p.conj(), axis=3).conj()
        m_min = -math.floor(f2.shape[3] / 2.0)
        m_max = math.ceil(f2.shape[3] / 2.0) - 1
        varphi = torch.linspace(-torch.pi, torch.pi, f2.shape[2]+ 1)[:-1]
        factor = torch.exp(-1j * varphi[None, :, None] * torch.arange(m_min, m_max+1)[None, None, :]).unsqueeze(0)
        f2f = f2 * factor
        f_hat = T1FFT.analyze(f2f.conj(), axis=2).conj()
        return f, f1c, f1p, f2, f2f, f_hat

    def synthesize(self, f_hat):
        f2f = T1FFT.synthesize(f_hat.conj(), axis=2).conj()
        m_min = -math.floor(f2f.shape[3] / 2.0)
        m_max = math.ceil(f2f.shape[3] / 2.0) - 1
        psi = torch.linspace(-torch.pi, torch.pi, f2f.shape[2]+1)[:-1]
        factor = torch.exp(1j * psi[:, None] * torch.arange(m_min, m_max + 1)[None, :])
        f2 = f2f * factor[None,None, ...]
        f1p = T1FFT.synthesize(f2.conj(), axis=3).conj()
        f1c = self.resample_p2c_3d(f1p)
        f = shift_ifft(f1c)
        return f, f1c, f1p, f2, f2f, f_hat

    def resample_c2p(self, fc):
        """
        Resample a function on a Cartesian grid to a polar grid.
        :param fc: function values sampled on a Cartesian grid.
        :return: resampled function on a polar grid
        """
        fp_r = map_coordinates_torch_constant(fc.real, self.c2p_coords)
        fp_c = map_coordinates_torch_constant(fc.imag, self.c2p_coords)
        fp = fp_r + 1j * fp_c
        return fp

    def resample_p2c(self, fp):
        """
        Resample a function on a polar grid to a Cartesian grid.
        :param fp: function values sampled on a polar grid.
        :return: resampled function on a Cartesian grid
        """
        fc_r = map_coordinates_torch_circular(fp.real,self.p2c_coords)
        fc_c = map_coordinates_torch_circular(fp.imag,self.p2c_coords)
        fc = fc_r + 1j * fc_c
        return fc

    def resample_c2p_3d(self, fc):
        """
        Resample a 3D function on a Cartesian grid to a polar grid.
        :param fc: 3D function values sampled on a Cartesian grid.
        :return: resampled function on a polar grid
        """
        if self.interpolation_method == 'spline':
            fp = [self.resample_c2p(fc[:, :, :, i]) for i in range(fc.shape[3])]
            return torch.stack(fp, dim=-1)

        elif self.interpolation_method == 'Fourier':
            fp = [self.flerp.forward(fc[:, :, i]) for i in range(fc.shape[2])]
            return torch.stack(fp, dim=-1)

    def resample_p2c_3d(self, fp):
        """
        Resample a 3D function on a polar grid to a Cartesian grid.
        :param fp: 3D function values sampled on a polar grid.
        :return: resampled function on a Cartesian grid
        """
        if self.interpolation_method == 'spline':
            fc = [self.resample_p2c(fp[:, :, :, i]) for i in range(fp.shape[3])]
            return torch.stack(fc, dim=-1)

        elif self.interpolation_method == 'Fourier':
            fc = [self.flerp.backward(fp[:, :, i]) for i in range(fp.shape[2])]
            return torch.stack(fc, dim=-1)
    
    def neg_log_likelihood(self, energy, pose: torch.Tensor) -> torch.Tensor:
        """
        Compute point-wise synthesize the SE2 Fourier transform M at a given pose.
        Args:
            eta (torch.Tensor): Fourier coefficients (eta) of SE2 distribution with shape [n, 3] where n is the number of
                                samples
            l_n_z (float): Log of normalization constant of SE2 distribution
            pose (torch.Tensor): Pose at which to interpolate the SE2 Fourier transform
            se2_fft: Object class for SE2 Fourier transform

        Returns:
            Probability of distribution determined by Fourier coefficients (moments) at given pose
        """
        # energy = torch.log(density+ 1e-40)
        l_n_z, _ = self.compute_moments_lnz(energy)
        _, _, _, _, _, eta = self.analyze(energy)
        b_x, b_y, b_t = self.spatial_grid_size
        # Reshape in case single pose is provided
        if pose.ndim < 2:
            pose = rearrange(pose, "b -> 1 b")
        # Arrange pose samples in broadcastable shape
        dx, dy = rearrange(pose[:,  0] + 0.5, "b -> b 1 1 ").to(energy.device), rearrange(pose[:, 1] + 0.5, "b -> b 1 1").to(energy.device)
        d_theta = rearrange(pose[:, 2] + math.pi, "b -> b 1 1 1 1").to(energy.device)
        # Synthesize signal to obtain first FFT
        _, _, _, f_p_psi_m, _, _ = self.synthesize(eta)
        # Shift the signal to the origin
        f_p_psi_m = rearrange(torch.fft.ifftshift(f_p_psi_m, dim=2), "b p n m -> b p n m 1")
        
        omega_n =  rearrange(torch.arange(b_t), "n -> 1 1 1 n 1").to(energy.device)
        # Compute the value of f(x) using the inverse Fourier transform
        f_p_psi = torch.sum(f_p_psi_m.conj().to(energy.device) * torch.exp(1j * omega_n * d_theta), dim=3).conj()
        # Map from polar to Cartesian grid
        f_p_p = self.resample_p2c_3d(f_p_psi)
        # Finally, 2D inverse FFT
        f_p_p = torch.fft.ifftshift(f_p_p,dim=(1,2,3)).to(energy.device)

        t_x, t_y = 1, 1
        # Compute complex term
        angle_x = (
            1j * 2 * torch.pi * (1 / t_x) * rearrange(torch.arange(b_x), "nx -> 1 nx 1 ").to(energy.device) * (dx)
        )  # Angle component in X
        angle_y = (
            1j * 2 * torch.pi * (1 / t_y) * rearrange(torch.arange(b_y), "ny -> 1 ny 1 ").to(energy.device) * (dy)
        )  # Angle component in Y
        
        angle = rearrange(angle_x, "b nx 1 -> b nx 1 1") + rearrange(
            angle_y, "b ny 1 -> b 1 ny 1"
        ).to(energy.device)
        f = -torch.sum(f_p_p * torch.exp(angle), dim=(1, 2, 3))
        # + l_n_z.to(energy.device)

        return (f + l_n_z.to(energy.device)).real
    
    def compute_moments_lnz(self, energy):
        """
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: moments and log partition constant
        """
        minimum= torch.min(torch.abs(energy)).to(energy.device)
        _ , _, _, _, _, unnormalized_moments_ = self.analyze(torch.exp(energy - minimum))
        density_ , _, _, _, _, _ = self.synthesize(unnormalized_moments_)
        _, _, _, _, _, unnormalized_moments = self.analyze(density_)
        z_0 = 0
        z_1 = unnormalized_moments.shape[2] // 2
        z_2 = unnormalized_moments.shape[3] // 2
        constant = (unnormalized_moments[:, z_0, z_1, z_2] * (math.pi * 2)).to(energy.device)
        l_n_z = torch.log(constant + 1e-8) + minimum
        # Update moments of distribution and constant only when needed
        return l_n_z.real, torch.exp(l_n_z).real