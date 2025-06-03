# Updated code with float64 precision

import torch
import torch.fft as fft
import torch.nn.functional as F
from src.distributions.SE2.T1FFT import T1FFT
from src.distributions.SE2.T2FFT import T2FFT
from src.utils.interpolation import bilinear_interpolate_torch_with_nan, bilinear_interpolate_torch_circular_padded
# from distributions.SE2.T1FFT import T1FFT
# from distributions.SE2.T2FFT import T2FFT
# from utils.interpolation import bilinear_interpolate_torch_with_nan, bilinear_interpolate_torch_circular_padded
from einops import rearrange
import math
import datetime
import pdb
from scipy.ndimage import map_coordinates
import numpy as np

def preprocess_input_with_cval(input, cval=float('nan')):
    N, C, H, W = input.shape
    extended = torch.full((N, C, H + 2, W + 2), cval, dtype=torch.float64, device=input.device)
    extended[:, :, 1:-1, 1:-1] = input
    return extended


def quadratic_bspline_kernel(x):
    abs_x = torch.abs(x)
    result = torch.where(
        abs_x < 0.5,
        0.75 - abs_x**2,
        torch.where(
            abs_x < 1.5,
            0.5 * (abs_x - 1.5)**2,
            torch.zeros_like(x)
        )
    )
    return result

def spline2d_interpolate(input, grid):
    """
    Approximates 2D order-2 (quadratic) B-spline interpolation using `grid_sample`.

    Args:
        input: (B, C, H, W) tensor.
        grid: (B, H_out, W_out, 2) sampling grid in [-1, 1] coordinates.
        padding_mode: 'zeros', 'border', or 'reflection'.

    Returns:
        Interpolated tensor of shape (B, C, H_out, W_out)
    """
    B, H, W = input.shape
    device = input.device

    wrapped_coords_x =  grid[1, ...] % input.shape[2]
    wrapped_coords_y = grid[0, ...] % input.shape[1]

    grid_x = wrapped_coords_x[..., 0] + 1
    grid_y = wrapped_coords_y[..., 1] + 1

    base_y = torch.floor(grid_y).long()
    base_x = torch.floor(grid_x).long()

    dy = grid_y - base_y.float()
    dx = grid_x - base_x.float()

    offsets = torch.tensor([-1, 0, 1], device=device)
    out = torch.zeros((B, grid.shape[1], grid.shape[2]), device=device)

    for oy in offsets:
        for ox in offsets:

            sampled = bilinear_interpolate_torch_circular_padded(input, wrapped_coords_x, wrapped_coords_y)

            # Compute weights using quadratic B-spline
            wy = quadratic_bspline_kernel(dy - oy)
            wx = quadratic_bspline_kernel(dx - ox)
            w = (wy * wx) # (B,  H_out, W_out)

            out += sampled * w

    return out


def adjust_grid_for_extension(grid):
    N, H, W, C = grid.shape
    adjusted_grid = grid.clone().to(torch.float64)
    adjusted_grid[..., 0] = 2.0 * adjusted_grid[..., 0] / (W - 1) - 1.0
    adjusted_grid[..., 1] = 2.0 * adjusted_grid[..., 1] / (H - 1) - 1.0
    return adjusted_grid
@torch.compile(fullgraph=False)
def map_coordinates_torch_circular_bilinear(input_tensor, coords):
    with torch.set_grad_enabled(input_tensor.requires_grad):  # Explicitly handle grad mode
        wrapped_coords_x = coords[1, ...] % input_tensor.shape[2]
        wrapped_coords_y = coords[0, ...] % input_tensor.shape[1]
        # wrapped_coords_x = coords[0, ...] % input_tensor.shape[1]
        # wrapped_coords_y = coords[1, ...] % input_tensor.shape[2]
        sampled = bilinear_interpolate_torch_circular_padded(input_tensor, wrapped_coords_x, wrapped_coords_y)
        return sampled
    
# @torch.compile(fullgraph=False)
# def map_coordinates_torch_circular_spline_order(input_tensor, coords, spline_order):
#     grid = torch.stack((coords[0], coords[1]), dim=-1)
#     grid_expand = grid.unsqueeze(0).expand(input_tensor.shape[0], -1, -1, -1)  # shape: [B, C, H, W]
#     sampled = interpol.grid_pull(input_tensor.unsqueeze(1), grid_expand, interpolation=spline_order, bound="wrap", prefilter=True)
#     return sampled.squeeze(1) 

# @torch.compile(fullgraph=False)
# def map_coordinates_torch_grid_sample_circular_spline_order(input_tensor, coords, spline_order):
#     # with torch.set_grad_enabled(input_tensor.requires_grad):  # Explicitly handle grad mode
#         # wrapped_coords_x = coords[1, ...] % input_tensor.shape[2]
#         # wrapped_coords_y = coords[0, ...] % input_tensor.shape[1]
#     # wrapped_coords_x = coords[1, ...] % input_tensor.shape[2]
#     # wrapped_coords_y = coords[0, ...] % input_tensor.shape[1]
#     wrapped_coords_x = coords[0, ...] % input_tensor.shape[1]
#     wrapped_coords_y = coords[1, ...] % input_tensor.shape[2]
#     grid = torch.stack((wrapped_coords_x, wrapped_coords_y), dim=-1) # shape: [H, W, 2]
#     grid_expand = grid.unsqueeze(0).expand(input_tensor.shape[0], -1, -1, -1)  # shape: [B, H, W, 2]
#     f_padded = F.pad(input_tensor, (1, 1, 1, 1), mode='replicate')
#     sampled = spline2d_interpolate(f_padded.unsqueeze(1), grid_expand)
#     # grid = torch.stack((wrapped_coords_x + 1, wrapped_coords_y + 1), dim=-1)
#     # grid_expand = grid.unsqueeze(0).expand(input_tensor.shape[0], -1, -1, -1)  # shape: [B, C, H, W]
#     # grid_expand_normalised = adjust_grid_for_extension(grid_expand)
#     # f_padded = F.pad(input_tensor, (1, 1, 1, 1), mode='replicate')
#     # sampled = F.grid_sample(f_padded.unsqueeze(1), grid_expand_normalised, mode='bicubic', padding_mode='border', align_corners=False)

#     return sampled.squeeze(1)  # Remove the channel dimension added by grid_pull

def map_coordinates_np(input_tensor, coords, spline_order):

    input_tensor_ = input_tensor.cpu().numpy()
    coords_ = coords.cpu().numpy()
    sampled = map_coordinates(input_tensor_, coords_, order=spline_order, mode='wrap')
    return sampled
    
@torch.compile(fullgraph=False)
def map_coordinates_torch_constant(input_tensor, coords):
     with torch.set_grad_enabled(input_tensor.requires_grad):
        fa = torch.empty((input_tensor.shape[0], input_tensor.shape[1] + 1, input_tensor.shape[2] + 1), dtype=torch.float64, device=input_tensor.device)
        fa[:, :-1, :-1] = input_tensor
        fa[:, -1, :-1] = input_tensor[:, 0, :]
        fa[:, :-1, -1] = input_tensor[:, :, 0]
        fa[:, -1, -1] = input_tensor[:, 0, 0]
        # wrapped_coords_x = coords[1, ...] % input_tensor.shape[2]
        # wrapped_coords_y = coords[0, ...] % input_tensor.shape[1]
        wrapped_coords_x = coords[0, ...] % input_tensor.shape[1]
        wrapped_coords_y = coords[1, ...] % input_tensor.shape[2]
        fa_padded = torch.nn.functional.pad(fa, (1, 1, 1, 1), mode='constant', value=float('nan'))
        sampled = bilinear_interpolate_torch_with_nan(fa_padded, wrapped_coords_x, wrapped_coords_y)
        return sampled

def map_coordinates_np_costant(input_tensor, coords):

    input_tensor_ = input_tensor[0].cpu().numpy()
    coords_ = coords.cpu().numpy()
    # Create an agumented array, where the last row and column are added at the beginning of the axes
    fa = np.empty((input_tensor_.shape[0] + 1, input_tensor_.shape[1] + 1))
    fa[:-1, :-1] = input_tensor_
    fa[-1, :-1] = input_tensor_[0, :]
    fa[:-1, -1] = input_tensor_[:, 0]
    fa[-1, -1] = input_tensor_[0, 0]

    # Wrap coordinates
    wrapped_coords_x = coords_[0, ...] % input_tensor_.shape[0]
    wrapped_coords_y = coords_[1, ...] % input_tensor_.shape[1]
    wrapped_coords = np.r_[wrapped_coords_x[None, ...], wrapped_coords_y[None, ...]]

    # Interpolate
    #return fa, wrapped_coords, map_coordinates(f, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)
    sampled = map_coordinates(fa, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)

    return torch.tensor(sampled).unsqueeze(0).to(input_tensor.device)  # Convert back to tensor and add batch dimension


def shift_fft(f):
    nx = f.shape[1]
    ny = f.shape[2]
    p0, q0 = nx // 2, ny // 2
    X, Y = torch.meshgrid(
        torch.arange(p0, p0 + nx, dtype=torch.float64) % nx,
        torch.arange(q0, q0 + ny, dtype=torch.float64) % ny,
        indexing="ij",
    )
    fs = f[:, X.long(), Y.long(), ...]
    return T2FFT.analyze(fs, axes=(1, 2))

def shift_ifft(fh):
    nx = fh.shape[1]
    ny = fh.shape[2]
    p0, q0 = nx // 2, ny // 2
    X, Y = torch.meshgrid(
        torch.arange(-p0, -p0 + nx, dtype=torch.float64) % nx,
        torch.arange(-q0, -q0 + ny, dtype=torch.float64) % ny,
        indexing="ij",
    )
    fs = T2FFT.synthesize(fh, axes=(1, 2))
    f = fs[:, X.long(), Y.long(), ...]
    return f



class SE2_FFT():
    def __init__(self, spatial_grid_size, device, interpolation_method="spline", spline_order=1, oversampling_factor=1):
        self.spatial_grid_size = spatial_grid_size
        self.interpolation_method = interpolation_method
        self.device = device

        if interpolation_method == "spline":
            self.spline_order = spline_order
            self.p0 = spatial_grid_size[0] // 2
            self.q0 = spatial_grid_size[1] // 2
            self.r_max = torch.sqrt(torch.tensor(self.p0**2 + self.q0**2, dtype=torch.float64, device=device))
            self.n_samples_r = int(oversampling_factor * (torch.ceil(self.r_max) + 1))
            self.n_samples_t = int(oversampling_factor * (torch.ceil(2 * torch.pi * self.r_max)))
            r = torch.linspace(0.0, self.r_max, self.n_samples_r, dtype=torch.float64, device=device)
            theta = torch.linspace(0, 2 * torch.pi, self.n_samples_t + 1, dtype=torch.float64, device=device)[:-1]
            R, THETA = torch.meshgrid(r, theta, indexing="ij")
            X = R * torch.cos(THETA)
            Y = R * torch.sin(THETA)
            I = X + self.p0
            J = Y + self.q0
            self.c2p_coords = torch.stack((I, J), dim=0)
            i = torch.arange(0, self.spatial_grid_size[0], dtype=torch.float64, device=device)
            j = torch.arange(0, self.spatial_grid_size[1], dtype=torch.float64, device=device)
            x = i - self.p0
            y = j - self.q0
            X, Y = torch.meshgrid(x, y, indexing="ij")
            R = torch.sqrt(X**2 + Y**2)
            # T = (torch.atan2(Y, X) + math.pi) % (2*math.pi)
            T = torch.atan2(Y, X)
            R *= (self.n_samples_r - 1) / self.r_max
            T *= self.n_samples_t / (2 * torch.pi)
            self.p2c_coords = torch.stack((R, T), dim=0)

        elif interpolation_method == "Fourier":
            r_max = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=torch.float64, device=device))
            nr = 15 * torch.ceil(r_max * spatial_grid_size[0])
            nt = 5 * torch.ceil(2 * torch.pi * r_max * spatial_grid_size[0])
            nx, ny = spatial_grid_size[:2]
            self.flerp = FourierInterpolator.init_cartesian_to_polar(nr, nt, nx, ny)
        else:
            raise ValueError("Unknown interpolation method:" + str(interpolation_method))

    def analyze(self, f):
        f1c = shift_fft(f).to(self.device)
        f1p = self.resample_c2p_3d(f1c)
        f2 = T1FFT.analyze(f1p.conj(), axis=3).conj()
        m_min = -math.floor(f2.shape[3] / 2.0)
        m_max = math.ceil(f2.shape[3] / 2.0) - 1
        varphi = torch.linspace(0, 2 * torch.pi, f2.shape[2] + 1, dtype=torch.float64, device=self.device)[:-1]
        factor = torch.exp(-1j * varphi[None, :, None] * torch.arange(m_min, m_max + 1, dtype=torch.float64, device=self.device)[None, None, :]).unsqueeze(0)
        f2f = f2 * factor
        f_hat = T1FFT.analyze(f2f.conj(), axis=2).conj()
        return f, f1c, f1p, f2, f2f, f_hat

    def synthesize(self, f_hat):
        f2f = T1FFT.synthesize(f_hat.conj(), axis=2).conj()
        m_min = -math.floor(f2f.shape[3] / 2.0)
        m_max = math.ceil(f2f.shape[3] / 2.0) - 1
        psi = torch.linspace(0, 2 * torch.pi, f2f.shape[2] + 1, dtype=torch.float64, device=self.device)[:-1]
        factor = torch.exp(1j * psi[:, None] * torch.arange(m_min, m_max + 1, dtype=torch.float64, device=self.device)[None, :])
        f2 = f2f * factor[None, None, ...]
        f1p = T1FFT.synthesize(f2.conj(), axis=3).conj().to(self.device)
        f1c = self.resample_p2c_3d(f1p)
        f = shift_ifft(f1c)
        # .permute(0, 2, 1, 3)
        return f, f1c, f1p, f2, f2f, f_hat

    def resample_c2p(self, fc):
        fp_r = map_coordinates_torch_constant(fc.real, self.c2p_coords)
        fp_c = map_coordinates_torch_constant(fc.imag, self.c2p_coords)
        fp = fp_r + 1j * fp_c
        return fp

    def resample_p2c(self, fp):
        if self.spline_order == 1:
            fc_r = map_coordinates_torch_circular_bilinear(fp.real, self.p2c_coords)
            fc_c = map_coordinates_torch_circular_bilinear(fp.imag, self.p2c_coords)
        elif self.spline_order == 2:
            fc_r = spline2d_interpolate(fp.real, self.p2c_coords)
            fc_c = spline2d_interpolate(fp.imag, self.p2c_coords)
        else:
            raise NotImplementedError(f"spline_order {self.spline_order} is not supported. Only orders 1 and 2 are implemented.")

        fc = fc_r + 1j * fc_c
        return fc 

    def resample_c2p_3d(self, fc):
        if self.interpolation_method == 'spline':
            fp = [self.resample_c2p(fc[:, :, :, i]) for i in range(fc.shape[3])]
            return torch.stack(fp, dim=-1)
        elif self.interpolation_method == 'Fourier':
            fp = [self.flerp.forward(fc[:, :, i]) for i in range(fc.shape[2])]
            return torch.stack(fp, dim=-1)

    def resample_p2c_3d(self, fp):
        if self.interpolation_method == 'spline':
            fc = [self.resample_p2c(fp[:, :, :, i]) for i in range(fp.shape[3])]
            return torch.stack(fc, dim=-1)
        elif self.interpolation_method == 'Fourier':
            fc = [self.flerp.backward(fp[:, :, i]) for i in range(fp.shape[2])]
            return torch.stack(fc, dim=-1)

    def neg_log_likelihood(self, energy, pose: torch.Tensor) -> torch.Tensor:
        l_n_z, _ = self.compute_moments_lnz(energy)
        _, _, _, _, _, eta = self.analyze(energy)
        b_x, b_y, b_t = self.spatial_grid_size
        if pose.ndim < 2:
            pose = rearrange(pose, "b -> 1 b")
        dx, dy = rearrange(pose[:, 0] + 0.5, "b -> b 1 1").to(self.device), rearrange(pose[:, 1] + 0.5, "b -> b 1 1").to(self.device)
        d_theta = rearrange(pose[:, 2] + torch.pi, "b -> b 1 1 1 1").to(self.device)
        _, _, _, f_p_psi_m, _, _ = self.synthesize(eta)
        f_p_psi_m = rearrange(torch.fft.ifftshift(f_p_psi_m, dim=2), "b p n m -> b p n m 1")
        omega_n = rearrange(torch.arange(b_t, dtype=torch.float64, device=self.device), "n -> 1 1 1 n 1")
        f_p_psi = torch.sum(f_p_psi_m.conj() * torch.exp(1j * omega_n * d_theta), dim=3).conj()
        f_p_p = self.resample_p2c_3d(f_p_psi)
        f_p_p = torch.fft.ifftshift(f_p_p, dim=(1, 2, 3))
        t_x, t_y = 1, 1
        angle_x = (
            1j * 2 * torch.pi * (1 / t_x) * rearrange(torch.arange(b_x, dtype=torch.float64, device=self.device), "nx -> 1 nx 1") * (dx)
        )
        angle_y = (
            1j * 2 * torch.pi * (1 / t_y) * rearrange(torch.arange(b_y, dtype=torch.float64, device=self.device), "ny -> 1 ny 1") * (dy)
        )
        angle = rearrange(angle_x, "b nx 1 -> b nx 1 1") + rearrange(
            angle_y, "b ny 1 -> b 1 ny 1"
        )
        f = -torch.sum(f_p_p * torch.exp(angle), dim=(1, 2, 3))
        return (f + l_n_z).real

    def compute_moments_lnz(self, energy):
        minimum = torch.min(torch.abs(energy)).to(self.device)
        _, _, _, _, _, unnormalized_moments = self.analyze(torch.exp(energy - minimum))
        z_0 = 0
        z_1 = unnormalized_moments.shape[2] // 2
        z_2 = unnormalized_moments.shape[3] // 2
        constant = (unnormalized_moments[:, z_0, z_1, z_2] * (math.pi * 2)).to(self.device)
        l_n_z = torch.log(constant + 1e-8) + minimum
        return l_n_z.real, torch.exp(l_n_z).real