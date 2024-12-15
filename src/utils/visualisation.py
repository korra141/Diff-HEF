import matplotlib.pyplot as plt
import math
import torch
import numpy as np
from scipy.special import i0
import os
import imageio
import re
import pdb
from scipy.stats import beta as beta_dist
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.distributions.S1.BetaDistribution import BetaDistribution

def fit_grid_into_larger(mean, range_x_diff, range_y_diff, band_limit, output_density):
    """
    Fits a smaller grid (output_density) into a larger grid (complete_grid) robustly.

    Args:
        mean (tuple): Center point (x, y) for the smaller grid.
        range_x_diff (float): Range of the smaller grid along the x-axis.
        range_y_diff (float): Range of the smaller grid along the y-axis.
        band_limit (tuple): Dimensions (height, width) of the smaller grid.
        output_density (np.ndarray): Density values for the smaller grid.

    Returns:
        np.ndarray: Updated larger grid with the smaller grid fitted in.
    """
    output_density = output_density.detach().numpy()
    # Step sizes
    step_x = range_x_diff / band_limit[0]
    step_y = range_y_diff / band_limit[1]

    # Initialize the larger grid
    grid_width = int(1 / step_x)  # Assumes range [0, 1] for larger grid
    grid_height = int(1 / step_y)
    complete_grid = np.ones((grid_height, grid_width)) * 1e-8

    # Compute starting indices for smaller grid
    index_x_start = int((mean[0] - range_x_diff / 2 + 0.5) / step_x)
    index_y_start = int((mean[1] - range_y_diff / 2 + 0.5) / step_y)

    # Compute end indices
    index_x_end = index_x_start + band_limit[0]
    index_y_end = index_y_start + band_limit[1]

    # Clip indices to ensure they stay within the bounds of the larger grid
    grid_x_start = max(0, index_x_start)
    grid_y_start = max(0, index_y_start)
    grid_x_end = min(grid_height, index_x_end)
    grid_y_end = min(grid_width, index_y_end)

    # Compute the region of the smaller grid that fits within the larger grid
    small_x_start = grid_x_start - index_x_start
    small_y_start = grid_y_start - index_y_start
    small_x_end = small_x_start + (grid_x_end - grid_x_start)
    small_y_end = small_y_start + (grid_y_end - grid_y_start)

    # Place the smaller grid into the larger grid
    complete_grid[
        grid_x_start:grid_x_end, grid_y_start:grid_y_end
    ] = output_density[
        small_x_start:small_x_end, small_y_start:small_y_end
    ]

    return torch.from_numpy(complete_grid)

def plot_circular_distribution(energy_samples,mean=None,legend="predicted",ax=None,range_theta=None):
    """
    Plots the distribution of data on a circle.

    Args:
      energy_samples: A tensor of energy samples.
    """
    if range_theta is None:
      L = 2*np.pi
    else :
      L = range_theta
    grid_size = energy_samples.shape[0]
    maximum = torch.max(energy_samples).unsqueeze(-1)
    moments = torch.fft.fft(torch.exp(energy_samples - maximum), dim=-1)
    ln_z_ = torch.log(L*moments[0] / grid_size).real.unsqueeze(-1) + maximum
    prob = torch.exp(energy_samples - ln_z_)
    prob = prob.detach()


    # Working on unit circle
    radii = 1.0

    # theta = torch.linspace(0,range_theta,grid_size)
    if range_theta is None:
      theta = torch.linspace(0,2*math.pi,grid_size)
    else :
      theta = torch.linspace(mean - range_theta/2 , mean + range_theta/2, grid_size)
    # theta = torch.linspace(mean,mean+range_theta,grid_size)
    # theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    theta_1 = torch.linspace(0, 2*math.pi, 100)
    theta_1 = torch.cat([theta_1, theta_1[0].unsqueeze(0)], 0)
    ct_1 = torch.cos(theta_1)
    st_1 = torch.sin(theta_1)

    if ax is None:
      fig, ax = plt.subplots(1, 1)

    # First plot circle
    ax.plot(ct_1, st_1, 'k-', lw=3, alpha=0.6)

    # Plot functions in polar coordinates
    # print(prob.shape)
    # prob = torch.cat([prob, prob[0, None]], 0)
    # Use only real components of the function and offset to unit radius
    prob_real = torch.real(prob) + radii
    f_x = ct * prob_real
    f_y = st * prob_real
    # Plot circle using x and y coordinates
    ax.plot(f_x, f_y, '-', lw=3, alpha=0.5,label=legend)
    # Only set axis off for polar plot
    plt.axis('off')
    # Set aspect ratio to equal, to create a perfect circle
    ax.set_aspect('equal')
    # Annotate axes in circle
    ax.text(1.05, 0, rf'0', style='italic', fontsize=15)
    ax.text(-1.15, 0, r'$\pi$', style='italic', fontsize=15)
    ax.text(0, 1.12, r'$\frac{\pi}{2}$', style='italic', fontsize=20)
    ax.text(0, -1.12, r'$3\frac{\pi}{2}$', style='italic', fontsize=20)
    return ax

def plot_s1_func(f, theta_new=None, legend=None, ax=None):

    # Working on unit circle
    radii = 1.0
    bandwidth = f.shape[0]
    # maximum = torch.max(energy_samples).unsqueeze(-1)
    moments = torch.fft.fft(f, dim=-1)
    z = (2*math.pi*moments[0] / bandwidth).real
    prob = f/z
    prob = prob.detach()



    # First plot the support of the distributions S^1
    tensor_start = torch.tensor(0, dtype=torch.float64)
    tensor_stop = torch.tensor(2 * math.pi, dtype=torch.float64)
    if theta_new is None:
      theta = torch.linspace(tensor_start, tensor_stop, bandwidth + 1)[:-1]
      theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
    else:
      theta = torch.linspace(tensor_start, tensor_stop, bandwidth + 1)[:-1]
      theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
      theta = (theta + theta_new) % (2 * math.pi)

    ct = torch.cos(theta)
    st = torch.sin(theta)

    theta_1 = torch.linspace(tensor_start, tensor_stop, 100)
    theta_1 = torch.concat([theta_1, theta_1[0, None]], 0)

    ct_1 = torch.cos(theta_1)
    st_1 = torch.sin(theta_1)

    # First plot circle
    ax.plot(ct_1, st_1, 'k-', lw=3, alpha=0.6)

    prob = torch.concat([prob, prob[0].unsqueeze(0)], 0)
    # Use only real components of the function and offset to unit radius
    f_real = prob.real + radii
    f_x = ct * f_real
    f_y = st * f_real
    # Plot circle using x and y coordinates
    ax.plot(f_x, f_y, '-', lw=3, alpha=0.5, label=legend)
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

def plot_beta_distribution(alpha, beta, theta_new, grid_size, ax, legend):
  """
  Plots a beta distribution on a circle.

  Args:
    alpha (float): Alpha parameter of the beta distribution.
    beta (float): Beta parameter of the beta distribution.
    grid_size (int): Number of points to plot.
    ax (matplotlib.axes.Axes): The axes to plot on.
    legend (str): Legend label for the plot.
  """

  theta = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
  if theta_new is not None:
    theta = (theta + theta_new) % (2 * np.pi)
  
  beta_dist = BetaDistribution(alpha, beta)
  beta_pdf = beta_dist.pdf(theta/(2 * np.pi))
  radius = 1.0

  prob_grid_r = beta_pdf + radius

  a = radius * np.cos(theta)
  b = radius * np.sin(theta)

  prob_grid_x = np.cos(theta) * prob_grid_r
  prob_grid_y = np.sin(theta) * prob_grid_r

  ax.plot(a, b)
  ax.plot(prob_grid_x, prob_grid_y, label=legend)
  ax.legend()

  return ax
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
    for filename in sorted(os.listdir(image_folder),key = lambda x : int(re.search(r'\d+', x).group())):
        if filename.endswith(".png") and (prefix is None or filename.startswith(prefix)): # Adjust file extension if needed
          image_path = os.path.join(image_folder, filename)
          images.append(imageio.imread(image_path))
    # pdb.set_trace()
    imageio.mimsave(gif_name, images,duration=duration)  # Adjust fps as needed

# # def plot_3d(ground_truth,measurement,range_x,range_y,band_limit,true_density, predicted_density,input_energy,folder_path, iteration=None, output=None):
#   x = torch.linspace(range_x[0], range_x[1], band_limit[0]+1)[:-1]
#   y = torch.linspace(range_y[0], range_y[1], band_limit[1]+1)[:-1]
#   x, y = torch.meshgrid(x, y)
#   x_numpy = x.numpy()
#   y_numpy = y.numpy()

#   # plotting_3d_density(x_numpy,y_numpy,predicted_density,true_density,folder_path)
#   fig, axs = plt.subplots(2, 2, figsize=(16, 16))
#   histogram_density(measurement, ground_truth, predicted_density, "measurement", axs[0,0])
#   histogram_density(measurement, ground_truth, input_energy, "prediction", axs[0,1])
#   histogram_density(measurement, ground_truth, true_density, "posterior", axs[1,1])
#   if output is not None:
#     histogram_density(measurement, ground_truth, output, "delta density predicted", axs[1,0])
  
#   plt.tight_layout()
#   if iteration is not None:
#     fig.suptitle(f"Analytical filter at iteration {iteration} plots")
#     plt.savefig(folder_path + f"/hef_2d_plots_{iteration}.png")
#   else:
#     plt.savefig(folder_path + f"/hef_2d_plots.png")
#   plt.close()

def plot_density(ground_truth,measurement,range_x,range_y,folder_path,dict_density,title,iter=None):

  values = list(dict_density.values())
  keys = list(dict_density.keys())
  # True and Predicted density
  plot_3d_density(range_x,range_y,values[0].detach(), values[1].detach(),folder_path,title)

  num_plots = len(values)
  num_rows = int(math.ceil(math.sqrt(num_plots)))
  num_cols = int(math.ceil(num_plots / num_rows))
  fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))

  for i, (key, value) in enumerate(dict_density.items()):
    row = i % num_rows
    col = i // num_cols
    if num_cols == 1:
      histogram_density(measurement, ground_truth, value.detach(), key, axs[row])
    else:
      histogram_density(measurement, ground_truth, value.detach(), key, axs[row, col])
  # # Hide any unused subplots
  for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flatten()[j])
  plt.tight_layout()
  fig.suptitle(title)
  if iter is not None:
    plt.savefig(folder_path + f"/2d_plots_{title}_{iter}.png",dpi=100)
  else:
    plt.savefig(folder_path + f"/2d_plots_{title}.png",dpi=100)
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

def plot_3d_density(range_x,range_y,true_density,predicted_density,folder_path,title,traj_iter=None):
  fig, axs = plt.subplots(2, figsize=(8, 8),subplot_kw={'projection': '3d'})

  n_samples_x, n_samples_y = true_density.shape
  x = np.linspace(range_x[0], range_x[1], n_samples_x +1)[:-1]
  y = np.linspace(range_y[0], range_y[1], n_samples_y +1)[:-1]
  x, y = np.meshgrid(x, y)
  temp_1 = axs[0].contour3D(x, y, true_density.numpy(), 50, cmap='viridis')

  n_samples_x, n_samples_y = predicted_density.shape
  x = np.linspace(range_x[0], range_x[1], n_samples_x +1)[:-1]
  y = np.linspace(range_y[0], range_y[1], n_samples_y +1)[:-1]
  x, y = np.meshgrid(x, y)
  temp = axs[1].contour3D(x, y, predicted_density.numpy(), 50, cmap='viridis')

  # Add labels and title
  axs[1].set_xlabel('x')
  axs[1].set_ylabel('y')
  axs[1].set_zlabel('Predicted distribution')

  axs[0].set_xlabel('x')
  axs[0].set_ylabel('y')
  axs[0].set_zlabel('True distribution')

  # Add a color bar to show the values of z
  fig.colorbar(temp, ax=axs[1], shrink=0.5, aspect=5,anchor=(0.5, 0.5))
  fig.colorbar(temp_1, ax=axs[0], shrink=0.5, aspect=5, anchor=(0.5, 0.5))
  

  if traj_iter is not None:
    fig.suptitle(f"Learning Filter Trajectory {traj_iter}")
    plt.savefig(folder_path + f"/plot3d_{title}_{traj_iter}.png")
  else:
    fig.suptitle(f"plot 3d {title}")
    plt.savefig(folder_path + f"/plot3d_{title}.png")
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

# def plot_3d_analytic_filter(ground_truth,measurement,range_x,range_y,band_limit,folder_path, dict_density,title,iter=None):
  # x = torch.linspace(range_x[0], range_x[1], band_limit[0]+1)[:-1]
  # y = torch.linspace(range_y[0], range_y[1], band_limit[1]+1)[:-1]
  # x, y = torch.meshgrid(x, y)
  # x_numpy = x.numpy()
  # y_numpy = y.numpy()  

  # values = list(dict_density.values())
  # keys = list(dict_density.keys())

  # fig, axs = plt.subplots(2, 3, figsize=(16, 16))
  # histogram_density(measurement, ground_truth, values[0].detach(), keys[0], axs[0,0])
  # histogram_density(measurement, ground_truth, values[1].detach(), keys[1], axs[0,1])
  # histogram_density(measurement, ground_truth, values[2].detach(), keys[2], axs[1,1])
  # if len(values) > 3:
  #   histogram_density(measurement, ground_truth, values[3].detach(), keys[3], axs[1,0])
  #   histogram_density(measurement, ground_truth, values[4].detach(), keys[4], axs[1,2])
  
  # plt.tight_layout()
  # fig.suptitle(title)
  # if iter is not None:
  #   plt.savefig(folder_path + f"/hef_2d_plots_{iter}.png")
  # else:
  #   plt.savefig(folder_path + f"/hef_2d_plots.png")
  # plt.close()