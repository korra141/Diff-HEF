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
from copy import deepcopy, copy
from matplotlib.lines import Line2D
from typing import Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("/home/mila/r/ria.arora/scratch/local/HarmonicExponentialBayesFitler/")
from src.distributions.S1.BetaDistribution import BetaDistribution
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(mu, cov, ax, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the distribution
    
    cov : torch.Tensor
        Covariance matrix of the distribution

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    ellise_color : Union[str, Tuple[float, float, float, float]]
        Color of the ellipse

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    # Convert tensors to numpy for matplotlib compatibility if needed
    if torch.is_tensor(mu):
        mu = mu.detach().cpu().numpy()
    if torch.is_tensor(cov):
        cov = cov.detach().cpu().numpy()

    pearson = cov[0, 1] / torch.sqrt(torch.tensor(cov[0, 0] * cov[1, 1]))
    if torch.is_tensor(pearson):
        pearson = pearson.item()
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = torch.sqrt(torch.tensor(1 + pearson)).item()
    ell_radius_y = torch.sqrt(torch.tensor(1 - pearson)).item()
    
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      edgecolor=kwargs['ellipse_color'],
                      label=kwargs['ellipse_label'],
                      facecolor="none",
                      lw=kwargs['lw'],
                      alpha=kwargs['alpha'])

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = torch.sqrt(torch.tensor(cov[0, 0])).item() * n_std
    mean_x = mu[0]

    # calculating the standard deviation of y ...
    scale_y = torch.sqrt(torch.tensor(cov[1, 1])).item() * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_confidence_ellipse(mu, cov, ax, n_std: list[float] = [1.0, 2.0], **kwargs) -> plt.Axes:
    """
    Plot confidence ellipse of a 2D Gaussian distribution
    Create a plot of the covariance confidence ellipse of `x` and `y` for multiple sigma values

    Parameters
    ----------
    mu : torch.Tensor
        Mean vector of the distribution
    
    cov : torch.Tensor
        Covariance matrix of the distribution

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : List[float]
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    # Convert tensors to numpy for matplotlib compatibility if needed
    if torch.is_tensor(mu):
        mu = mu.detach().cpu().numpy()
    if torch.is_tensor(cov):
        cov = cov.detach().cpu().numpy()
        
    # Plot confidence ellipses at different level
    n = len(n_std)
    cmap = kwargs.pop('cmap')
    
    # Plot mean of the distribution
    ax.scatter(mu[0], mu[1], **kwargs)

    # Plot confidence ellipses
    colors = cmap(torch.linspace(0, 1, n + 2).numpy())
    for i, std in enumerate(n_std):
        kwargs['ellipse_color'] = colors[i]
        kwargs['ellipse_label'] = rf'{std}$\sigma$'
        confidence_ellipse(mu, cov, ax, n_std=std, **kwargs)
        
    kwargs.pop('ellipse_color')
    kwargs.pop('ellipse_label')
    return ax

def plot_se2_contours(fs: list[torch.Tensor],
            x: torch.Tensor,
            y: torch.Tensor,
            theta: torch.Tensor,
            level_contours: bool = True,
            titles: list[str] = None,
            config: Optional[list[Tuple[str, str, Optional[str]]]] = None):
  """
  Plots functions on SE(2).

  Functions are represented by a sampled grid f.
  If f is 3 dimensional (x, y, theta), theta is marginalized out
  :param fs: List of functions to plot
  :param x: x values of the grid
  :param y: y values of the grid
  :param theta: theta values of the grid
  :param level_contours: Boolean flag to add level contours on main plot
  :param titles: List of titles for each subplot
  :param config: Dict with **kwargs for plotting
  :return ax: List of axes of the plot
  """

  # First three axis are prior, measurement and posterior. Fourth one is standard contour plot
  fig = plt.figure(constrained_layout=True, figsize=(12, 9))
  gs = fig.add_gridspec(3, 4)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[2, 0])
  # ax4 = fig.add_subplot(gs[1, 1])
  # ax5 = fig.add_subplot(gs[2, 0])
  ax4 = fig.add_subplot(gs[:, 1:])
  # Zip first three axes into a list
  axes = [ax1, ax2, ax3]

  # gs = fig.add_gridspec(3, 2)
  # ax1 = fig.add_subplot(gs[0, 0])
  # ax2 = fig.add_subplot(gs[0, 1])
  # ax3 = fig.add_subplot(gs[1, 0])
  # ax4 = fig.add_subplot(gs[1, 1])
  # ax5 = fig.add_subplot(gs[2, 0])
  # # ax4 = fig.add_subplot(gs[:, 1:])
  # # Zip first three axes into a list
  # axes = [ax1, ax2, ax3,ax4, ax5]

  if titles is None:
    titles = [rf'$f_{i}$' for i, _ in enumerate(fs)]

  if x.ndim == 3:
    x = x[:, :, 0]
    y = y[:, :, 0]

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  color_maps = [plt.cm.Purples, plt.cm.Reds, plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

  # pdb.set_trace()
  # Dynamically choose best level for the contour plot
  # level = torch.min(torch.tensor([torch.trapz(fs[i], x=theta, dim=-1).max() for i in range(len(fs))])) / 2
  level = 0.5
  legend_elements = []

  for i, f in enumerate(fs):
    if f.ndim == 3:
      f = torch.trapz(f, x=theta, dim=2)
    
    # pdb.set_trace()
    f_scaled = (f - f.min()) / (f.max() - f.min())

    # Color mesh plot of the distributions
    axes[i].pcolormesh(x.cpu().numpy(), y.cpu().numpy(), f_scaled.cpu().numpy(), shading='auto', cmap=color_maps[i], vmin=0, vmax=1.0)
    axes[i].set_title(titles[i])
    # Whether to plot or not contours in main plot
    if level_contours:
      cp = ax4.contour(x.cpu().numpy(), y.cpu().numpy(), f_scaled.cpu().numpy(), levels=[level], colors=colors[i], linewidths=2)
      legend_elements.append(cp.legend_elements()[0][0])

  proxy = []
  if level_contours:
    proxy = [Line2D([0], [0], color=e.get_color(), lw=2, label=titles[i]) for i, e in enumerate(legend_elements)]
  # Extra legend elements
  if config is not None:
    # Make a copy as this will remove some elements
    temp = deepcopy(config)
    for params in temp:
      if params.get('s'):
        params.pop('s')
      if params.get('cmap'):
        params.pop('cmap')
      proxy.extend([Line2D([], [], linestyle='none', **params)])

  ax4.set_title(f'Probability contour at {np.round(level, 2)}')
  ax4.legend(handles=proxy, fancybox=True, framealpha=1, shadow=True, borderpad=1)
  # # Append contour axis to axes
  axes.append(ax4)

  return axes


def plot_se2_mean_filters(fs: list[torch.Tensor],
              x: torch.Tensor,
              y: torch.Tensor,
              theta: torch.Tensor,
              samples: torch.Tensor,
              iteration: int,
              level_contours: bool = True,
              contour_titles: list[str] = None,
              config: Optional[list[Tuple[str, str, Optional[str]]]] = None,
              beacons: Optional[torch.Tensor] = None):
  """
  Plots functions on SE(2).

  Functions are represented by a sampled grid f.
  If f is 3 dimensional (x, y, theta), theta is marginalized out
  :param fs: List of functions to plot
  :param x: x values of the grid
  :param y: y values of the grid
  :param theta: theta values of the grid
  :param samples: mean estimate baseline filters, assume last element dimension are landmarks
  :param iteration: current iteration
  :param level_contours: Boolean flag to add level contours on main plot
  :param contour_titles: List of titles for each contourplot's subplot
  :param config: Dict with **kwargs for plotting
  :param beacons: Tensor with ground truth beacons
  :return ax: List of axes of the plot
  """
  # Plot contours
  cfg = deepcopy(config)
  add_beacons = beacons is not None
  axes = plot_se2_contours(fs, x.clone(), y.clone(), theta.clone(),
               level_contours, contour_titles, cfg if add_beacons else cfg[:-1])
  # for i, (key, value) in enumerate(samples.items()):
    # Get config for current samples

  axis_index_points = len(fs)

  for i, (key,value) in enumerate(samples.items()):
    for c in cfg:
      if c.get('label') == key:
        break
      # If correct config wasn't found, go on to next series.
      if c.get('label') != key:
        continue
      # Remove unneeded keys
    c['edgecolor'] = c.pop('markeredgecolor')
    # Get latest sample
    point = value.clone()
    axes[axis_index_points].scatter(point[0].item(), point[1].item(), **c)

  # Plot beacons
  if add_beacons:
    plot_beacons(beacons, axes[axis_index_points])
  return axes


def plot_beacons(beacons: torch.Tensor, ax: plt.Axes, color: str = 'dimgrey', marker: str = 'o'):
  ax.scatter(beacons[:, 0].cpu().numpy(), beacons[:, 1].cpu().numpy(), c=color, marker=marker, alpha=1.0, s=150, edgecolor='k', linewidths=1)



def plot_se2_filters(filters: dict[str, list[np.ndarray]],
                     x: np.ndarray,
                     y: np.ndarray,
                     theta: np.ndarray,
                     beacons: np.ndarray,
                     titles: list[str],
                     config: list[str]):
    """
    Plot the current estimate of different filters on SE(2)

    The result of each filter is a tuple of (mean, covariance/samples) which are used to represent its uncertainty. For
    each plot, the heading is marginalized
    :param filters: Dictionary where each key contains a filter (e.g., HEF, EKF, PF, HistF) in a list. The first element
    is its mean and the second element is its covariance/samples. It is possible to add not only samples but other
    values which will be plotted as well such as ground truth.
    :param x: x values of the grid
    :param y: y values of the grid
    :param theta: angles of the grid
    :param beacons: beacons to plot
    :param titles: Title for each plot, should be of size 4, one for each filter
    :param config: List of extra legend and param entries, should be of size 4, one for each filter
    :return ax: List of axes of the plot
    """
    cfg = deepcopy(config)
    # Each axis correspond to a different filter
    fig = plt.figure(constrained_layout=True, figsize=(9, 9))
    # gs = fig.add_gridspec(3, 3)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])
    # ax5 = fig.add_subplot(gs[0, 2])
    # ax6 = fig.add_subplot(gs[1, 2])
    # ax7 = fig.add_subplot(gs[2, 0])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    # ax4 = fig.add_subplot(gs[:, 1:])
    axes = [ax1, ax2, ax3, ax4]
    
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    if x.ndim == 3:
        x = x[:, :, 0]
        y = y[:, :, 0]

    ### Plot Harmonic filter ###
    for c in cfg:
        if c.get('label') == 'HEF':
            break
    c.pop('label')
    c['edgecolor'] = c.pop('markeredgecolor')
    cmap = c.pop('cmap')
    c["label"] = "Mean"
    harmonic = filters['HEF']
    # Plot mean
    ax1.scatter(harmonic[0][0], harmonic[0][1], **c)
    # c["label"] = "Mode"
    # c["edgecolor"] = "honeydew"
    # c["linewidth"] = 1.5
    # ax1.scatter(harmonic[2][0], harmonic[2][1], **c)
    if harmonic[1].ndim == 3:
        harmonic_posterior = np.trapz(harmonic[1], x=theta, axis=2)
    # Scale distribution between 0 - 1
    max_value, min_value = harmonic_posterior.max(), harmonic_posterior.min()
    scaled_posterior = (harmonic_posterior - min_value) / (max_value - min_value)
    ax1.pcolormesh(x, y, scaled_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)

    ### Plot Diff EKF ###
    # for c in cfg:
    #     if c.get('label') == 'Diff-EKF':
    #         break
    # c.pop('label')
    # c['edgecolor'] = c.pop('markeredgecolor')
    # diff_ekf = filters['Diff-EKF']
    # c["label"] = "Mean"
    # plot_confidence_ellipse(diff_ekf[0][:2], diff_ekf[1][0:2, 0:2], ax5, **c)

    ### Plot EKF ###
    for c in cfg:
      if c.get('label') == 'EKF':
          break
    c.pop('label')
    c['edgecolor'] = c.pop('markeredgecolor')
    ekf = filters['EKF']
    c["label"] = "Mean"
    plot_confidence_ellipse(ekf[0][:2], ekf[1][0:2, 0:2], ax2, **c)

    ### PLot LSTM ###
    # for c in cfg:
    #   if c.get('label') == 'LSTM':
    #       break
    # c.pop('label')
    # c['edgecolor'] = c.pop('markeredgecolor')
    # lstm = filters['LSTM']
    # c["label"] = "Mean"
    # plot_confidence_ellipse(lstm[0][:2], lstm[1][0:2, 0:2], ax7, **c)

    ### Plot PF ###
    for c in cfg:
        if c.get('label') == 'PF':
            break
    c.pop('label')
    c['label'] =  "Mean"
    pf = filters['PF']
    c['edgecolor'] = c.pop('markeredgecolor')
    # Select randomly a percentage of particles to plot
    percentage = 0.125 / 2
    n_particles = pf[1].shape[0]
    n_particles_to_plot = int(n_particles * percentage)
    idx = np.random.choice(n_particles, n_particles_to_plot, replace=False)
    ax4.scatter(pf[0][0], pf[0][1], **c)
    # c["label"] = "Mode"
    # c["edgecolor"] = "honeydew"
    # c["linewidth"] = 1.5
    # ax3.scatter(pf[2][0], pf[2][1], **c)
    ax4.scatter(pf[1][idx, 0], pf[1][idx, 1], c=c['c'], s=30, alpha=0.2, marker=c['marker'], zorder=0)

    # ### Plot HF ###
    for c in cfg:
        if c.get('label') == 'HistF':
            break
    c.pop('label')
    cmap = c.pop('cmap')
    c["label"] = "Mean"
    c['edgecolor'] = c.pop('markeredgecolor')
    hf = filters['HistF']
    ax3.scatter(hf[0][0], hf[0][1], **c)
    # c["label"] = "Mode"
    # c["edgecolor"] = "honeydew"
    # c["linewidth"] = 1.5
    # ax4.scatter(hf[2][0], hf[2][1], **c)
    hf_posterior = hf[1].sum(-1)
    max_value, min_value = hf_posterior.max(), hf_posterior.min()
    hf_posterior = (hf_posterior - min_value) / (max_value - min_value)
    ax3.pcolormesh(x, y, hf_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)

    ### Plot Diff-Hist ###
    # for c in cfg:
    #     if c.get('label') == 'Diff-HistF':
    #         break
    # c.pop('label')
    # cmap = c.pop('cmap')
    # c["label"] = "Mean"
    # c['edgecolor'] = c.pop('markeredgecolor')
    # diff_hf = filters['Diff-HistF']
    # ax6.scatter(diff_hf[0][0], diff_hf[0][1], **c)
    # # c["label"] = "Mode"
    # # c["edgecolor"] = "honeydew"
    # # c["linewidth"] = 1.5
    # # ax4.scatter(hf[2][0], hf[2][1], **c)
    # diff_hf_posterior = diff_hf[1].sum(-1)
    # max_value, min_value = diff_hf_posterior.max(), diff_hf_posterior.min()
    # diff_hf_posterior = (diff_hf_posterior - min_value) / (max_value - min_value)
    # ax6.pcolormesh(x, y, diff_hf_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)

    ### Plot beacons and ground truth ###
    for c in cfg:
        if c.get('label') == 'GT':
            break
    c['edgecolor'] = c.pop('markeredgecolor')
    gt_pose = filters['GT']
    # Add a line plot in for loop
    for i, ax in enumerate(axes):
        # Plot ground truth
        ax.scatter(gt_pose[0][0], gt_pose[0][1], **c)
        # Plot beacons
        plot_beacons(beacons, ax)
        ax.set_title(titles[i], fontdict={'fontsize': 18})
        ax.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize="8")
        ax.set_aspect("equal")

    return axes


def plot_range_simulator_se2(range_data, text_labels, ax=None):
  """
  Plots the range simulator for SE(2) with text annotations.

  Args:
    range_data (list of tuples): List of (x, y, theta) tuples representing the SE(2) poses.
    text_labels (list of str): List of text labels corresponding to each pose.
    ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.

  Returns:
    matplotlib.axes.Axes: The axes with the plot.
  """
  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 8))

  for (x, y, theta), label in zip(range_data, text_labels):
    # Plot the position
    ax.plot(x, y, 'o', label=label)
    # Plot the orientation as an arrow
    dx = 0.1 * np.cos(theta)
    dy = 0.1 * np.sin(theta)
    ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    # Add text label
    ax.text(x + 0.05, y + 0.05, label, fontsize=10, color='red')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title('Range Simulator for SE(2)')
  ax.set_aspect('equal')
  ax.grid(True)
  return ax

def plot_distributions_s1(measurement_noise, band_limit, energy, ground_truth, measurements, epoch, i, j, logging_path):
  fig, ax = plt.subplots()
  measurements = measurements.cpu()
  ground_truth = ground_truth.cpu()
  ax = plot_circular_distribution(energy,legend="predicted distribution",ax=ax)
  ax = plotting_von_mises(ground_truth[j], measurement_noise**2, 100, ax, "true distribution")
  ax.plot(torch.cos(measurements[j]), torch.sin(measurements[j]), 'o', label="measurement data")
  ax.plot(torch.cos(ground_truth[j]), torch.sin(ground_truth[j]), 'o', label="pose data")
  ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
  ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
  ax.set_aspect('equal')
  plt.savefig(os.path.join(logging_path, f"training_s1_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
  plt.close(fig)

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
    prob = prob.detach().cpu()


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
    theta_1 = torch.linspace(0, 2*math.pi, 100).to(energy_samples.device).cpu()
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

# def plot_s1_func(f, theta_new=None, legend=None, ax=None):

#     # Working on unit circle
#     radii = 1.0
#     bandwidth = f.shape[0]
#     # maximum = torch.max(energy_samples).unsqueeze(-1)
#     moments = torch.fft.fft(f, dim=-1)
#     z = (2*math.pi*moments[0] / bandwidth).real
#     prob = f/z
#     prob = prob.detach()



#     # First plot the support of the distributions S^1
#     tensor_start = torch.tensor(0, dtype=torch.float64)
#     tensor_stop = torch.tensor(2 * math.pi, dtype=torch.float64)
#     if theta_new is None:
#       theta = torch.linspace(tensor_start, tensor_stop, bandwidth + 1)[:-1]
#       theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
#     else:
#       theta = torch.linspace(tensor_start, tensor_stop, bandwidth + 1)[:-1]
#       theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
#       theta = (theta + theta_new) % (2 * math.pi)

#     ct = torch.cos(theta)
#     st = torch.sin(theta)

#     theta_1 = torch.linspace(tensor_start, tensor_stop, 100)
#     theta_1 = torch.concat([theta_1, theta_1[0, None]], 0)

#     ct_1 = torch.cos(theta_1)
#     st_1 = torch.sin(theta_1)

#     # First plot circle
#     ax.plot(ct_1, st_1, 'k-', lw=3, alpha=0.6)

#     prob = torch.concat([prob, prob[0].unsqueeze(0)], 0)
#     # Use only real components of the function and offset to unit radius
#     f_real = prob.real + radii
#     f_x = ct * f_real
#     f_y = st * f_real
#     # Plot circle using x and y coordinates
#     ax.plot(f_x, f_y, '-', lw=3, alpha=0.5, label=legend)
#     # Only set axis off for polar plot
#     plt.axis('off')
#     # Set aspect ratio to equal, to create a perfect circle
#     ax.set_aspect('equal')
#     # Annotate axes in circle
#     ax.text(1.05, 0, rf'0', style='italic', fontsize=15)
#     ax.text(-1.15, 0, r'$\pi$', style='italic', fontsize=15)
#     ax.text(0, 1.12, r'$\frac{\pi}{2}$', style='italic', fontsize=20)
#     ax.text(0, -1.12, r'$-\frac{\pi}{2}$', style='italic', fontsize=20)
#     return ax


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

def plot_beta_distribution(beta_pdf, grid_size, ax, legend):
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
  # if theta_new is not None:
  #   theta = (theta + theta_new) % (2 * np.pi)
  
  # beta_dist = BetaDistribution(alpha, beta)
  # beta_pdf = beta_dist.pdf(theta/(2 * np.pi))
  radius = 1.0

  prob_grid_r = beta_pdf.detach().numpy() + radius

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
    kappa = 1 / cov

    theta = np.linspace(0, 2 * np.pi, grid_size+1)[:-1]
    vmf = np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))
    radius = 1.0

    prob_grid_r = vmf + radius

    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    prob_grid_x = np.cos(theta) * prob_grid_r
    prob_grid_y = np.sin(theta) * prob_grid_r

    plt.axis('off')
    # Set aspect ratio to equal, to create a perfect circle
    ax.set_aspect('equal')
    ax.plot(a, b, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.plot(prob_grid_x, prob_grid_y, label=legend, linewidth=2, alpha=0.8)

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
  im = ax.imshow(normalised_density.detach().cpu().numpy().T, cmap='viridis', origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])
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
  temp_1 = axs[0].contour3D(x, y, true_density.cpu().numpy(), 50, cmap='viridis')

  n_samples_x, n_samples_y = predicted_density.shape
  x = np.linspace(range_x[0], range_x[1], n_samples_x +1)[:-1]
  y = np.linspace(range_y[0], range_y[1], n_samples_y +1)[:-1]
  x, y = np.meshgrid(x, y)
  temp = axs[1].contour3D(x, y, predicted_density.cpu().numpy(), 50, cmap='viridis')

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


if __name__ == "__main__":
  # Example usage of plot_se2_contours
  grid_size = (100, 100, 100)
  # x = torch.linspace(-0.5, 0.5, grid_size[0])
  # y = torch.linspace(-0.5, 0.5, grid_size[1])
  # theta = torch.linspace(-math.pi, math.pi, grid_size[2])
  # x, y, theta = torch.meshgrid(x, y, theta)

  # poses = torch.stack((x.flatten(), y.flatten(), theta.flatten()), dim=-1).unsqueeze(0)
  # mu = torch.tensor([0, 0, math.pi]).unsqueeze(0)
  # cov = torch.diag(torch.tensor([0.1, 0.1, 0.1]))

  # f = GaussianDistribution_se2(mu,cov, grid_size)
  # fs = f.density(poses).reshape(-1, *grid_size)

  xs = np.linspace(-0.5, 0.5, grid_size[0],endpoint=False)
  ys = np.linspace(-0.5, 0.5, grid_size[1],endpoint=False)
  ts = np.linspace(-np.pi, np.pi, grid_size[2],endpoint=False)
  X, Y, T = np.meshgrid(xs, ys, ts)
  Poses = np.vstack((X.flatten(), Y.flatten(), T.flatten()))
  fft = SE2_FFT(spatial_grid_size=grid_size,interpolation_method='spline', spline_order=1, oversampling_factor=1)

  mu = np.array([0, 0, np.pi])
  cov = np.diag([0.1, 0.1, 0.1])
  gaussian = SE2Gaussian(mu, cov,samples=Poses.T,fft=fft)
  f = gaussian.prob.real
  


  print("Plotting contours for SE(2) functions")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  f = torch.tensor(f, device=device)
  X = torch.tensor(X, device=device)
  Y = torch.tensor(Y, device=device)
  T = torch.tensor(T, device=device)

  # Call the function
  # axes = plot_se2_contours([fs[0]], x, y, theta)
  axes = plot_se2_contours([f.reshape(grid_size)], X, Y,T)
  print("Plotting")
  # Show the plot
  path = os.path.join(os.getcwd(), "test_fig_torch_1.png")
  plt.savefig(path)
  print("Show")