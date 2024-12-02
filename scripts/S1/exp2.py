"""
This file contains an experiment to test whether its possible to learn the measueremnt distribution in S1 using
the Harmonic Exponential Functions. There is no process model and the measurement distribution is learned using 
the true position as input. 

Caveats:
- Since there are multiple trajectories that overlap each other, there is additional ambiguity about which distribution
the measurements are coming from. This is expected to increase the variance of the learned distribution or in the best case
scenario lead to multiple modes in the learned distribution.


Same as Experiment 1 but with Heteroscedastic noise.
"""


import time
import torch
import argparse
import torch.fft
import torch.nn as nn
import torch.optim as optim
from scipy.special import i0

import math
import os   

import matplotlib.pyplot as plt
import numpy as np
from time import strftime, localtime

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment to learn the measurement distribution in S1 using Harmonic Exponential Functions.')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=10, help='Number of neurons in the hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--band_limit', type=int, default=100, help='Band limit')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--measurement_noise_min', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--measurement_noise_max', type=float, default=0.3, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument('--shuffle_flag', type=bool, default=True, help='Whether to shuffle the samples')
    parser.add_argument("--log-dir", type=str, default="./logs/exp2/", help="Directory to save the logs")    
    parser.add_argument("--log-freq", type=int, default=10, help="Frequency of logging the results")
    return parser.parse_args()  

args = parse_args()

def heteroscedastic_noise(x, min_noise, max_noise):
   # Normalize x 
   x = x % (2 * np.pi)
   x = x / (2 * np.pi)
   scale = x**2 / (x**2 + (1-x)**2)
   return min_noise + scale * (max_noise - min_noise)


def generating_data_S1(args, batch_size, n_samples, trajectory_length, measurement_noise, step=0.1, shuffle_flag=True):
  """Generates training data for a system with circular motion.

  Args:
    batch_size: The number of samples in each batch.
    n_samples: The total number of samples to generate.
    trajectory_length: The length of each trajectory.
    measurement_noise: The standard deviation of the measurement noise.
    step: The step between poses in trajectory
    shuffle_flag: Whether to shuffle the samples,

  Returns:
    Flattened pose and noisy measurement data in a TensorDataset.
  """

  starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
  true_trajectories = np.ndarray((n_samples, trajectory_length))
  measurements = np.ndarray((n_samples, trajectory_length))

  for i in range(n_samples):
    # Generate a circular trajectory with a random starting position.
    initial_angle = starting_positions[i]
    trajectory = initial_angle + np.arange(trajectory_length) * step
    true_trajectories[i] = trajectory

    # Add Gaussian noise to the measurements.
    noise = [np.random.normal(0, heteroscedastic_noise(x, args.measurement_noise_min, args.measurement_noise_max)) for x in trajectory]
    measurements[i] = (trajectory + noise) % (2 * np.pi)

  measurements_ = torch.from_numpy(measurements % (2 * np.pi))
  ground_truth_ = torch.from_numpy(true_trajectories % (2 * np.pi))
  ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
  measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
  train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_flag)
  return train_loader


def compute_coefficients(energy):
  eta = torch.fft.fft(energy)
  eta = torch.fft.fftshift(eta)
  return eta


def compute_normalization_constant(energy, band_limit):
  # print(energy.size())
  max_energy, _ = torch.max(energy, axis=1)
  # print(max_energy)
  moment = torch.fft.fft(torch.exp(energy - max_energy.unsqueeze(-1))) # Taking the FFT of the energy

  lnZ = torch.abs(torch.log(moment[:, 0] / (torch.pi ** 2 * band_limit / 62)) + max_energy) # log of the normalization constant
  return lnZ

def negative_loglikelihood(energy, measurement, band_limit):
  eta = compute_coefficients(energy)
  freq = torch.range(0, band_limit-1)
  # shifting the freq
  freq = freq - math.floor(band_limit/2)

  # Calculating the unnormalized likelihood
  # exponent = 1j * (torch.dot(measurement.reshape(-1, 1), freq.unsqueeze(0)))
  exponent = 1j * (freq.unsqueeze(0) * measurement.reshape(-1, 1))

  loglikelihood = torch.sum(eta * torch.exp(exponent), axis=-1) / band_limit

  # Normalizing the likelihood
  lnZ = compute_normalization_constant(energy, band_limit)
  loglikelihood = (loglikelihood - lnZ)
  # print(torch.mean(lnZ))

  return - torch.abs(torch.sum(loglikelihood)/energy.size(0))



def loss_fn(energy, measurements, grid_size=20):
  """Computes the loss function for the circular motion model.

  This function calculates the loss based on the negative log likelihood of the predicted distribution with the ground truth noisy measurements value
  Args:
    mu: The mean angle of the predicted distribution.
    cov: The covariance of the predicted distribution.
    measurements: The observed measurements.
    grid_size: The number of grid points to use for calculating the energy.

  Returns:
    ln_z_: The computed loss value.
  """
  # energy = compute_energy(mu, cov, grid_size)
  eta = torch.fft.fftshift(torch.fft.fft(energy, dim=-1), dim=-1)
  maximum = torch.max(energy, dim=-1).values.unsqueeze(-1)
  moments = torch.fft.fft(torch.exp(energy - maximum), dim=-1)
  ln_z_ = torch.log(moments[:, 0] / (math.pi * grid_size * math.pi / 62)).real.unsqueeze(-1) + maximum

  # taking inverse FFT over a set of frequencies ranging
  k_values = torch.arange(grid_size) - grid_size / 2
  k_values = k_values.unsqueeze(0).unsqueeze(-1)  # [1, num_samples, 1]
  value = measurements.unsqueeze(1)  # [batch_size, 1, 1]
  exponential_term = torch.exp(1j * k_values * value)  # [batch_size, num_samples, 1]

  inverse_transform = (eta.unsqueeze(-1) * exponential_term).sum(dim=1).real # [batch_size, 1]
  return torch.mean(-inverse_transform/grid_size + ln_z_, axis=0)

def plot_circular_distribution(energy_samples,legend="predicted",ax=None):
    """
    Plots the distribution of data on a circle.

    Args:
      energy_samples: A tensor of energy samples.
    """
    grid_size = energy_samples.shape[0]
    maximum = torch.max(energy_samples).unsqueeze(-1)
    moments = torch.fft.fft(torch.exp(energy_samples - maximum), dim=-1)
    ln_z_ = torch.log(moments[0] / (math.pi * grid_size * math.pi / 62)).real.unsqueeze(-1) + maximum
    prob = torch.exp(energy_samples - ln_z_)
    prob = prob.detach()


    # Working on unit circle
    radii = 1.0

    theta = torch.linspace(0, 2*math.pi, grid_size)
    theta = torch.cat([theta, theta[0].unsqueeze(0)], 0)
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
    prob = torch.cat([prob, prob[0, None]], 0)
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
    ax.text(0, -1.12, r'$-\frac{\pi}{2}$', style='italic', fontsize=20)
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
    ax.plot(prob_grid_x, prob_grid_y,label=legend[0])
    # ax.set_xlabel('Theta')
    # ax.set_ylabel('Density')
    ax.legend()

    return ax

# Define the neural network architecture
class EnergyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def main(args):

    # Generate training data
    train_loader =  generating_data_S1(args, args.batch_size, args.n_samples, args.trajectory_length, args.measurement_noise_min, args.step_size, True)
    test_loader = generating_data_S1(args, args.batch_size, 20, args.trajectory_length, args.measurement_noise_min, args.step_size, False)

    # Initialize the model, optimizer, and loss function
    model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))

    os.makedirs(os.path.join(args.log_dir, str(ctime)))


    # Training loop
    loss_list = []
    for epoch in range(args.num_epochs):
        if epoch % args.log_freq == 0:
            os.makedirs(os.path.join(args.log_dir, f"{ctime}/{epoch}/"))

        loss_tot = 0
        for i, (ground_truth, measurements) in enumerate(train_loader):
            # print(torch.cat([ground_truth, measurements], axis=-1))
            # break
            # Forward pass
            energy = model(ground_truth)
            loss = loss_fn(energy, measurements, args.band_limit)
            if i % 50 == 0 and epoch % args.log_freq == 0:
                fig, ax = plt.subplots()
                ax = plot_circular_distribution(energy[0],"predicted distribution",ax)
                ax = plotting_von_mises(ground_truth[0],heteroscedastic_noise(ground_truth[0], args.measurement_noise_min, args.measurement_noise_max).item()**2, args.band_limit,ax,"true distribution")
                ax.plot(torch.cos(measurements[0]),torch.sin(measurements[0]),'o',label="measurement data")
                ax.plot(torch.cos(ground_truth[0]), torch.sin(ground_truth[0]), 'o', label="pose data")
                ax.set_title(f"Epoch {epoch,}", loc='center')
                ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
                plt.savefig(os.path.join(args.log_dir, f"{ctime}/{epoch}/iter {i}.png"), format='png', dpi=300)
                # plt.show()
                plt.close()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tot += loss.item()




        # if (i+1) % 10 == 0:
        loss_list.append(loss_tot/len(train_loader))
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_tot/len(train_loader):.4f}')
    plt.plot(range(args.num_epochs),loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(args.log_dir, f"{ctime}/loss.png"), format='png', dpi=300)
    plt.close()
    print("Training finished!")


if __name__ == '__main__':
   args = parse_args()
   main(args)