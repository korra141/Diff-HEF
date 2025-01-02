import torch

class T2FFT():
    """
    The Fast Fourier Transform on the 2-Torus.

    The torus is parameterized by two cyclic variables (x, y).
    The standard domain is (x, y) in [0, 1) x [0, 1), in which case the Fourier basis functions are:
     exp( i 2 pi xi^T (x; y))
    where xi is the spectral variable, xi in Z^2.

    The Fourier transform is
    \hat{f}[p, q] = 1/2pi int_0^2pi f(x, y) exp(-i 2 pi xi^T (x; y)) dx dy

    but this class allows one to use arbitrarily scaled and shifted domains D = [l_x, u_x) x [l_y, u_y)
    Let the width of the domain be given by
      alpha_x = u_x - l_x
      alpha_y = u_y - l_y
    The basis functions on [l_x, u_x) x [l_y, u_y) are
     exp( i 2 pi xi^T ((x - l_x) / alpha_x; (y - l_y) / alpha_y))
    where xi is the spectral variable, xi in Z^2.
    The normalized Haar measure is dx dy / (alpha_x * alpha_y) (in terms of Lebesque measure dx dy)

    So the Fourier transform on this particular parameterization of the torus is:
    \hat{f}_pq = 1/alpha int_lx^ux int_ly^uy f(x) e^{-2 pi i (p, q)^T ((x - lx) / alpha_x; (y - ly)/alpha_y)} dx dy

    This is what the current class computes, given discrete samples in the domain D.
    The samples are assumed to come from the following sampling grid:
    (x_i, y_j), i = 0, ... N - 1; j = 0, ..., N - 1
    x_i = lx + alpha_x * (i / N_x)
    y_i = ly + alpha_y * (i / N_y)
    this is the output of
    x = torch.linspace(lx, ux, N_x, endpoint=False)
    y = torch.linspace(ly, uy, N_y, endpoint=False)
    X, Y = torch.meshgrid(x, y)

    """
    def __init__(self, lower_bound=(0., 0.), upper_bound=(1., 1.)):
        self.lower_bound = torch.tensor(lower_bound)
        self.upper_bound = torch.tensor(upper_bound)

    @staticmethod
    def analyze(f, axes=(0, 1)):
        """
        Compute the Fourier Transform of the discretely sampled function f : T^2 -> C.

        :param f: Input function sampled on a regular grid.
        :param axes: Axes along which to compute the FFT.
        :return: Fourier coefficients.
        """
        # Perform FFT
        f_hat = torch.fft.fft2(f, dim=axes)
        # Shift zero frequency component to the center
        f_hat = torch.fft.fftshift(f_hat, dim=axes)
        # Normalize
        size = torch.prod(torch.tensor([f.shape[ax] for ax in axes], dtype=torch.float32))
        return f_hat / size

    @staticmethod
    def synthesize(f_hat, axes=(0, 1)):
        """
        :param f_hat: Fourier coefficients.
        :param axes: Axes along which to compute the IFFT.
        :return: Reconstructed function.
        """
        # Compute the size of the input
        size = torch.prod(torch.tensor([f_hat.shape[ax] for ax in axes], dtype=torch.float32))
        # Shift zero frequency component back to original position
        f_hat = torch.fft.ifftshift(f_hat * size, dim=axes)
        # Perform inverse FFT
        f = torch.fft.ifft2(f_hat, dim=axes)
        return f
