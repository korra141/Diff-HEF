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

def generate_gif(image_folder, gif_name,prefix=None,duration=1):
    """Generates a GIF from a folder of images.

    Args:
    image_folder: The path to the folder containing the images.
    gif_name: The name of the output GIF file.
    """
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png") and (prefix is None or filename.startswith(prefix)): # Adjust file extension if needed
          image_path = os.path.join(image_folder, filename)
          images.append(imageio.imread(image_path))
    # pdb.set_trace()
    imageio.mimsave(gif_name, images,duration=duration)  # Adjust fps as needed

def plot_3d(ground_truth,measurement,range_x,range_y,band_limit,true_density, predicted_density,input_energy,folder_path, iteration, output=None):
  x = torch.linspace(range_x[0], range_x[1], band_limit[0]+1)[:-1]
  y = torch.linspace(range_y[0], range_y[1], band_limit[1]+1)[:-1]
  x, y = torch.meshgrid(x, y)
  x_numpy = x.numpy()
  y_numpy = y.numpy()

  # plotting_3d_density(x_numpy,y_numpy,predicted_density,true_density,folder_path)
  fig, axs = plt.subplots(2, 2, figsize=(16, 16))
  histogram_density(measurement, ground_truth, predicted_density, "measurement", axs[0,0])
  histogram_density(measurement, ground_truth, input_energy, "prediction", axs[0,1])
  histogram_density(measurement, ground_truth, true_density, "posterior", axs[1,1])
  if output is not None:
    histogram_density(measurement, ground_truth, output, "delta density predicted", axs[1,0])
  
  plt.tight_layout()
  fig.suptitle(f"Analytical filter at iteration {iteration} plots")
  plt.savefig(folder_path + f"/hef_2d_plots_{iteration}.png")
  plt.close()

def plot_3d_density(ground_truth,measurement,range_x,range_y,band_limit,folder_path, dict_density,title):
  x = torch.linspace(range_x[0], range_x[1], band_limit[0]+1)[:-1]
  y = torch.linspace(range_y[0], range_y[1], band_limit[1]+1)[:-1]
  x, y = torch.meshgrid(x, y)
  x_numpy = x.numpy()
  y_numpy = y.numpy()  

  values = list(dict_density.values())
  keys = list(dict_density.keys())
  # True and Predicted density
  plotting_3d_density(x_numpy,y_numpy,values[0].detach(), values[1].detach(),folder_path)

  fig, axs = plt.subplots(2, 2, figsize=(16, 16))
  histogram_density(measurement, ground_truth, values[0].detach(), keys[0], axs[0,0])
  histogram_density(measurement, ground_truth, values[1].detach(), keys[1], axs[0,1])
  histogram_density(measurement, ground_truth, values[2].detach(), keys[2], axs[1,1])
  
  plt.tight_layout()
  fig.suptitle(title)
  plt.savefig(folder_path + f"/2d_plots.png")
  plt.close()

def histogram_density(measurements_2d, pose_2d, normalised_density, legend, ax):
# Create a 2D histogram of the output energy distribution
  im = ax.imshow(normalised_density.detach().numpy().T, cmap='viridis', origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])
  ax.set_title(legend)
  ax.scatter(measurements_2d[0].item(), measurements_2d[1].item(), color='red', label='Measurement', s=50)
  ax.scatter(pose_2d[0].item(), pose_2d[1].item(), color='blue', label='Pose', s=50)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.legend()
  plt.colorbar(im, ax=ax, label='Density')

def plotting_3d_density(x,y,true_density,predicted_density,folder_path,traj_iter=None):
  fig, axs = plt.subplots(2, figsize=(8, 8),subplot_kw={'projection': '3d'})

  temp = axs[1].contour3D(x, y, predicted_density.numpy(), 50, cmap='viridis')
  temp_1 = axs[0].contour3D(x, y, true_density.numpy(), 50, cmap='viridis')

  # Add labels and title
  axs[1].set_xlabel('x')
  axs[1].set_ylabel('y')
  axs[1].set_zlabel('Predicted distribution')

  axs[0].set_xlabel('x')
  axs[0].set_ylabel('y')
  axs[0].set_zlabel('True distribution')

  # Add a color bar to show the values of z
  fig.colorbar(temp, ax=axs[0], shrink=0.5, aspect=5)
  fig.colorbar(temp_1, ax=axs[0], shrink=0.5, aspect=5)
  

  if traj_iter is not None:
    fig.suptitle(f"Learning Filter Trajectory {traj_iter}")
    plt.savefig(folder_path + f"/plot3d_{traj_iter}.png")
  else:
    plt.savefig(folder_path + "/plot3d.png")
  plt.close()

def plot_gaussian_energy(energy):
  """
  Plots a 3D contour of the given energy values using a Gaussian distribution.
  Parameters:
  energy (torch.Tensor): A 2D tensor containing the energy values to be plotted.
  This function creates a 3D contour plot of the energy values provided. It uses a Gaussian distribution
  to visualize the energy in a 3D space. The plot includes labeled axes and a color bar to indicate the 
  values of the z-axis.
  Note:
  This function can help plot energy to debug.
  """

  fig, axs = plt.subplots(1, figsize=(10, 8),subplot_kw={'projection': '3d'})
  x = np.linspace(-0.5, 0.5, band_limit[0],endpoint=False)
  y = np.linspace(-0.5, 0.5, band_limit[1],endpoint=False)
  x, y = np.meshgrid(x, y)
  # density = torch.exp(energy) / torch.sum(torch.exp(energy))
  temp = axs.contour3D(x, y, energy.numpy(), 50, cmap='viridis')
  # temp_1 = axs[0,1].contour3D(x_numpy, y_numpy, true_density.numpy(), 50, cmap='viridis')

  # Add labels and title
  axs.set_xlabel('x')
  axs.set_ylabel('y')
  axs.set_zlabel('Predicted distribution')

  # axs[0,1].set_xlabel('x')
  # axs[0,1].set_ylabel('y')
  # axs[0,1].set_zlabel('True distribution')

  # histogram_density(measurement, ground_truth_2d[0],normalised_density[0],"temp",ax=ax)
  # histogram_density(measurement, ground_truth_2d[0],normalised_density[0],"temp",ax=ax)


  # Add a color bar to show the values of z
  fig.colorbar(temp, ax=axs, shrink=0.5, aspect=5)
  # fig.colorbar(temp_1, ax=axs[1], shrink=0.5, aspect=5)

  # Show the plot
  plt.show()

def plot_3d_filter(ground_truth,measurement,range_x,range_y,band_limit, true_density, predicted_density, predicted_belief, posterior, folder_path,traj_iter):
  x = torch.linspace(range_x[0], range_x[1], band_limit[0]+1)[:-1]
  y = torch.linspace(range_y[0], range_y[1], band_limit[1]+1)[:-1]
  x, y = torch.meshgrid(x, y)
  x_numpy = x.numpy()
  y_numpy = y.numpy()

  predicted_density = predicted_density.detach()
  predicted_belief = predicted_belief.detach()
  posterior = posterior.detach()

  plotting_3d_density(x_numpy,y_numpy,true_density,predicted_density,folder_path,traj_iter)
  fig, axs = plt.subplots(2, 3, figsize=(16, 16))

  histogram_density(measurement, ground_truth, predicted_density, "measurement", axs[0,0])
  histogram_density(measurement, ground_truth, true_density, "analytic filter measurement model", axs[0,1])
  # histogram_density(measurement, ground_truth, process, "process", axs[0,2])
  histogram_density(measurement, ground_truth, predicted_belief, "predicted_belief", axs[1,0])
  # histogram_density(measurement, ground_truth, true_density, "true_density", axs[1,1])
  histogram_density(measurement, ground_truth, posterior, "posterior", axs[1,2])
  

  fig.suptitle(f"Learning Filter Trajectory {traj_iter}")
  plt.tight_layout()
  plt.savefig(folder_path + f"/2d_plots_{traj_iter}.png")
  plt.close()