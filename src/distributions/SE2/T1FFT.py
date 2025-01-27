import torch

class T1FFT():
    """
    The Fast Fourier Transform on the Circle / 1-Torus / 1-Sphere.
    """

    @staticmethod
    def analyze(f, axis=0):
        """
        Compute the Fourier Transform of the discretely sampled function f : T^1 -> C.

        :param f: Input function sampled on a regular grid.
        :param axis: Axis along which to compute the FFT.
        :return: Fourier coefficients.
        """
        # Perform FFT
        fhat = torch.fft.fft(f, dim=axis)
        # Shift zero frequency component to the center
        fhat = torch.fft.fftshift(fhat, dim=axis)
        # Normalize
        return fhat/f.shape[axis]

    @staticmethod
    def synthesize(f_hat, axis=0):
        """
        Compute the inverse / synthesis Fourier transform of the function f_hat : Z -> C.

        :param f_hat: Fourier coefficients.
        :param axis: Axis along which to compute the IFFT.
        :return: Reconstructed function.
        """
        # Shift zero frequency component back to original position
        f_hat = torch.fft.ifftshift(f_hat * f_hat.shape[axis], dim=axis)
        f = torch.fft.ifft(f_hat, dim=axis)
        return f

    @staticmethod
    def analyze_naive(f):
        """
        Naive implementation of the Fourier Transform.

        :param f: Input function sampled on a regular grid.
        :return: Fourier coefficients.
        """
        N = f.size(0)
        f_hat = torch.zeros_like(f, dtype=torch.complex64)
        for n in range(N):
            for k in range(N):
                theta_k = k * 2 * torch.pi / N
                f_hat[n] += f[k] * torch.exp(-1j * n * theta_k)
        return torch.fft.fftshift(f_hat / N, dim=0)