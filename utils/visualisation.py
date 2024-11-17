import matplotlib.pyplot as plt
import math
import torch
import numpy as np
from scipy.special import i0
import os
import imageio
import pdb

def plot_s1_func(f, legend=None, ax=None, plot_type: str = 'polar'):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if legend is None:
        legend = [rf'$prob_{i}$' for i, _ in enumerate(f)]

    # Working on unit circle
    radii = 1.0
    bandwidth = f[0].shape[0]

    # First plot the support of the distributions S^1
    tensor_start = torch.tensor(0, dtype=torch.float64)
    tensor_stop = torch.tensor(2 * math.pi, dtype=torch.float64)
    theta = torch.linspace(tensor_start, tensor_stop, bandwidth + 1)[:-1]
    theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)

    ct = torch.cos(theta)
    st = torch.sin(theta)

    theta_1 = torch.linspace(tensor_start, tensor_stop, 100)
    theta_1 = torch.concat([theta_1, theta_1[0, None]], 0)

    ct_1 = torch.cos(theta_1)
    st_1 = torch.sin(theta_1)

    # First plot circle
    ax.plot(ct_1, st_1, 'k-', lw=3, alpha=0.6)

    # Plot functions in polar coordinates
    for i, f_bar in enumerate(f):
        # Concat first element to close function
        # pdb.set_trace()
        f_bar = torch.concat([f_bar, f_bar[0].unsqueeze(0)], 0)
        # Use only real components of the function and offset to unit radius
        f_real = f_bar.real + radii
        f_x = ct * f_real
        f_y = st * f_real
        # Plot circle using x and y coordinates
        ax.plot(f_x, f_y, '-', lw=3, alpha=0.5, label=legend[i])
    # Only set axis off for polar plot
    plt.axis('off')
    # Set aspect ratio to equal, to create a perfect circle
    ax.set_aspect('equal')
    # Annotate axes in circle
    ax.text(1.05, 0, rf'0', style='italic', fontsize=15)
    ax.text(-1.15, 0, r'$\pi$', style='italic', fontsize=15)
    ax.text(0, 1.12, r'$\frac{\pi}{2}$', style='italic', fontsize=20)
    ax.text(0, -1.12, r'$-\frac{\pi}{2}$', style='italic', fontsize=20)
    return ax


def plot_s1_energy(energy_samples_list,
                   legend=None,
                   ax=None,
                   plot_type: str = 'polar'):
    """Process multiple functions at once for plotting"""

    f = []
    for energy_samples in energy_samples_list:
        grid_size = energy_samples.shape[0]
        maximum = torch.max(energy_samples)
        moments = torch.fft.rfft(torch.exp(energy_samples - maximum))
        ln_z_ = torch.log(moments[0] / (math.pi * grid_size * math.pi / 62)).real + maximum
        prob = torch.exp(energy_samples - ln_z_)
        f.append(prob)
    return plot_s1_func(f, legend, ax, plot_type)

def plotting_von_mises(mu,cov,grid_size,ax,legend):

    # pdb.set_trace()
    mu = mu.item()
    cov = cov
    kappa = 1 / cov

    theta = np.linspace(0, 2 * np.pi, grid_size+1)[:-1]
    vmf = np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))
    radius = 1.0

    prob_grid_r = vmf + radius

    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    prob_grid_x = np.cos(theta) * prob_grid_r
    prob_grid_y = np.sin(theta) * prob_grid_r

    ax.plot(a, b)
    ax.plot(prob_grid_x, prob_grid_y,label=legend)
    ax.legend()

    return ax

def generate_gif(image_folder, gif_name):
    """Generates a GIF from a folder of images.

    Args:
    image_folder: The path to the folder containing the images.
    gif_name: The name of the output GIF file.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):  # Adjust file extension if needed
          image_path = os.path.join(image_folder, filename)
          images.append(imageio.imread(image_path))
    # pdb.set_trace()
    imageio.mimsave(gif_name, images)  # Adjust fps as needed