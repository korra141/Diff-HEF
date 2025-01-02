import torch
import torch.fft as fft
import torch.nn.functional as F
from .T1FFT import T1FFT
from .T2FFT import T2FFT
from einops import rearrange


def shift_fft(f):
    nx, ny = f.shape[:2]
    p0, q0 = nx // 2, ny // 2

    X, Y = torch.meshgrid(
        torch.arange(p0, p0 + nx) % nx,
        torch.arange(q0, q0 + ny) % ny,
        indexing="ij",
    )

    fs = f[X, Y]
    return T2FFT.analyze(fs, axes=(0, 1))

def shift_ifft(fh):
    nx, ny = fh.shape[:2]
    p0, q0 = nx // 2, ny // 2

    X, Y = torch.meshgrid(
        torch.arange(-p0, -p0 + nx) % nx,
        torch.arange(-q0, -q0 + ny) % ny,
        indexing="ij",
    )

    fs = T2FFT.synthesize(fh, axes=(0, 1))
    f = fs[X, Y]
    return f

def cartesian_grid(nx, ny):
    x = torch.linspace(-0.5, 0.5, nx, endpoint=False)
    y = torch.linspace(-0.5, 0.5, ny, endpoint=False)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X, Y


class SE2_FFT(FFTBase):
    def __init__(
        self,
        spatial_grid_size=(10, 10, 10),
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

            r = torch.linspace(0.0, self.r_max, self.n_samples_r, endpoint=True)
            theta = torch.linspace(0, 2 * torch.pi, self.n_samples_t, endpoint=False)
            R, THETA = torch.meshgrid(r, theta, indexing="ij")

            X = R * torch.cos(THETA)
            Y = R * torch.sin(THETA)

            I = X + self.p0
            J = Y + self.q0

            self.c2p_coords = torch.stack((I, J), dim=0)

            i = torch.arange(0, self.spatial_grid_size[0])
            j = torch.arange(0, self.spatial_grid_size[1])
            x = i - self.p0
            y = j - self.q0
            X, Y = torch.meshgrid(x, y, indexing="ij")

            R = torch.sqrt(X**2 + Y**2)
            T = torch.atan2(Y, X)

            R *= (self.n_samples_r - 1) / self.r_max
            T *= self.n_samples_t / (2 * torch.pi)

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

        f2 = T1FFT.analyze(f1p.conj(), axis=2).conj()

        m_min = -torch.floor(f2.shape[2] / 2.0)
        m_max = torch.ceil(f1p.shape[2] / 2.0) - 1
        varphi = torch.linspace(0, 2 * torch.pi, f2.shape[1], endpoint=False)
        factor = torch.exp(-1j * varphi[None, :, None] * torch.arange(m_min, m_max + 1)[None, None, :])
        f2f = f2 * factor

        f_hat = T1FFT.analyze(f2f.conj(), axis=1).conj()
        return f, f1c, f1p, f2, f2f, f_hat

    def synthesize(self, f_hat):
        f2f = T1FFT.synthesize(f_hat.conj(), axis=1).conj()

        m_min = -torch.floor(f2f.shape[2] / 2.0)
        m_max = torch.ceil(f2f.shape[2] / 2.0) - 1
        psi = torch.linspace(0, 2 * torch.pi, f2f.shape[1], endpoint=False)
        factor = torch.exp(1j * psi[:, None] * torch.arange(m_min, m_max + 1)[None, :])

        f2 = f2f * factor[None, ...]
        f1p = T1FFT.synthesize(f2.conj(), axis=2).conj()

        f1c = self.resample_p2c_3d(f1p)
        f = shift_ifft(f1c)

        return f, f1c, f1p, f2, f2f, f_hat

    def resample_c2p(self, fc):
        fp_r = F.grid_sample(fc.real.unsqueeze(0).unsqueeze(0), self.c2p_coords, mode="bilinear", align_corners=False)
        fp_c = F.grid_sample(fc.imag.unsqueeze(0).unsqueeze(0), self.c2p_coords, mode="bilinear", align_corners=False)
        fp = fp_r + 1j * fp_c
        return fp.squeeze()

    def resample_p2c(self, fp):
        fc_r = F.grid_sample(fp.real.unsqueeze(0).unsqueeze(0), self.p2c_coords, mode="bilinear", align_corners=False)
        fc_c = F.grid_sample(fp.imag.unsqueeze(0).unsqueeze(0), self.p2c_coords, mode="bilinear", align_corners=False)
        fc = fc_r + 1j * fc_c
        return fc.squeeze()

    def resample_c2p_3d(self, fc):
        if self.interpolation_method == "spline":
            return torch.stack([self.resample_c2p(fc[:, :, i]) for i in range(fc.shape[2])], dim=-1)
        elif self.interpolation_method == "Fourier":
            return torch.stack([self.flerp.forward(fc[:, :, i]) for i in range(fc.shape[2])], dim=-1)

    def resample_p2c_3d(self, fp):
        if self.interpolation_method == "spline":
            return torch.stack([self.resample_p2c(fp[:, :, i]) for i in range(fp.shape[2])], dim=-1)
        elif self.interpolation_method == "Fourier":
            return torch.stack([self.flerp.backward(fp[:, :, i]) for i in range(fp.shape[2])], dim=-1)

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
        l_n_z = compute_moments_lnz(energy)
        _, _, _, _, _, eta = self.analyze(energy)
        b_x, b_y, b_t = self.spatial_grid_size
        # Reshape in case single pose is provided
        if pose.ndim < 2:
            pose = rearrange(pose, "b -> 1 b")
        # Arrange pose samples in broadcastable shape
        dx, dy = rearrange(pose[:, 0], "b -> 1 b"), rearrange(pose[:, 1], "b -> 1 b")
        d_theta = rearrange(pose[:, 2], "b -> 1 b")
        # Synthesize signal to obtain first FFT
        _, _, _, f_p_psi_m, _, _ = self.synthesize(eta)
        # Shift the signal to the origin
        f_p_psi_m = rearrange(torch.fft.ifftshift(f_p_psi_m, dim=2), "p n m -> p n m 1")
        # Theta ranges from 0 to 2pi, thus ts = 2 * np.pi (duration)
        t_theta = 2 * torch.pi
        # Evaluate Fourier coefficients at desired point
        omega_n = (
            2 * torch.pi * (1 / t_theta) * rearrange(torch.arange(b_t), "n -> 1 1 n 1")
        )
        # Compute the value of f(x) using the inverse Fourier transform
        f_p_psi = torch.sum(f_p_psi_m * torch.exp(1j * omega_n * d_theta), dim=2)
        # Map from polar to Cartesian grid
        f_p_p = se2_fft.resample_p2c_3d(f_p_psi)
        # Finally, 2D inverse FFT
        f_p_p = shift_ifft(f_p_p)
        # Set domain of X and Y, recall X and Y range from [-0.5, 0.5]
        t_x, t_y = 1.0, 1.0
        # Compute complex term
        angle_x = (
            1j * 2 * torch.pi * (1 / t_x) * rearrange(torch.arange(b_x), "nx -> nx 1") * (dx + 0.5)
        )  # Angle component in X
        angle_y = (
            1j * 2 * torch.pi * (1 / t_y) * rearrange(torch.arange(b_y), "ny -> ny 1") * (dy + 0.5)
        )  # Angle component in Y
        angle = rearrange(angle_x, "nx b -> nx 1 b") + rearrange(
            angle_y, "ny b -> 1 ny b"
        )
        # Compute the value of log(p(g)) using the inverse Fourier transform
        f = torch.sum(f_p_p * torch.exp(angle), dim=(0, 1)).real
        return f
    

    def compute_moments_lnz(self, negative_energy):
        """
        Compute the moments of the distribution
        :param eta: canonical parameters in log prob space of distribution
        :param update: whether to update the moments of the distribution and log partition constant
        :return: moments and log partition constant
        """
        # negative_energy, _, _, _, _, _ = self.synthesize(eta)
        maximum = torch.max(negative_energy)
        _, _, _, _, _, unnormalized_moments = self.analyze(torch.exp(negative_energy - maximum))
        # TODO: Figure out why z_0 is the 0 index.
        z_0 = 0
        z_1 = unnormalized_moments.shape[1] // 2
        z_2 = unnormalized_moments.shape[2] // 2
        # Scale by invariant haar measure
        # Haar measure in Chirikjian's book
        # unnormalized_moments[z_0, z_1, z_2] *= np.power(2 * np.pi, 2)
        # Haar measure in Chirikjian's book for S1, semi-direct product R^2 \cross S1
        unnormalized_moments[z_0, z_1, z_2] *= math.pi * 2
        moments = unnormalized_moments / unnormalized_moments[z_0, z_1, z_2]
        moments[z_0, z_1, z_2] = unnormalized_moments[z_0, z_1, z_2]
        l_n_z = np.log(unnormalized_moments[z_0, z_1, z_2]) + maximum
        # Update moments of distribution and constant only when needed
        return moments, l_n_z.real