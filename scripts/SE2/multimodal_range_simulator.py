import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import sys
import math
import datetime
import random
import pdb
base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)

import pdb
torch.cuda.empty_cache()
# from src.utils.visualisation import plot_density, plot_distributions_s1
# from src.distributions.R2.StandardDistribution import GaussianDistribution as GaussianDistribution_r2
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.SE2.GaussianDistribution import MultiModalGaussianSE2 as MultiModal_se2
from src.distributions.SE2.SE2_FFT import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from src.filter.HEF_SE2 import HEFilter
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean

# parameters
NUM_TRAJECTORIES = 1
TRAJECTORY_LENGTH = 80
STEP_MOTION = SE2Group(0.01, 0.00, np.pi / 40)
MOTION_NOISE = np.sqrt(np.array([0.001, 0.001, 0.001]))
MEASUREMENT_NOISE = np.sqrt(0.0001)
batch_size = 1
validation_split = 0.2
grid_size = (50, 50, 32)
mu_prior = torch.tensor([[0.0, -0.15, 0], [0.0, 0.15, 0]])
cov_prior = torch.tile(torch.diag(torch.tensor([0.1, 0.1, 0.1])),(2,1,1))
range_x = (-0.5, 0.5)
range_y = (-0.5, 0.5)
poses, X, Y, T = se2_grid_samples_torch(batch_size ,grid_size)
# Set seed for reproducibility
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# Generate dataset
train_data_loader = generate_bounded_se2_dataset(
    num_trajectories=NUM_TRAJECTORIES,
    trajectory_length=TRAJECTORY_LENGTH,
    step_motion=STEP_MOTION,
    motion_noise=MOTION_NOISE,
    measurement_noise=MEASUREMENT_NOISE,
    samples = poses.numpy(),
    start_pose = SE2Group(0.0, -0.15, 0),
)
# Initialize the HEF filter
hef_filter = HEFilter(grid_size,range_x,range_y)
# Initialize the SE2 FFT
fft = SE2_FFT(spatial_grid_size=grid_size,
                  interpolation_method='spline',
                  spline_order=1,
                  oversampling_factor=1)

prior_pdf = MultiModal_se2(mu_prior, cov_prior,grid_size).density(poses).reshape(-1,*grid_size)
trajectory_list = []
mean_nll = 0
for j, data in enumerate(train_data_loader):
    inputs, measurements, control = data
    for i in range(TRAJECTORY_LENGTH):
        # Perform operations on inputs and labels using HEF analytical filter here
        process = GaussianDistribution_se2(control[:, i], torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float32),grid_size)
        process_pdf  = process.density(poses).reshape(-1, *grid_size)
        eta_bel_x_t_bar, density_bel_x_t_bar = hef_filter.predict(prior_pdf, process_pdf)   
        density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32)

        _, _, _, _, _, eta = fft.analyze(measurements[:, i].reshape(-1, *grid_size))
        measurement_energy, _, _, _, _, eta = fft.synthesize(eta)
        measurement_energy = measurement_energy.reshape(-1, *grid_size)
        lnz , z_m = fft.compute_moments_lnz(measurement_energy)
        measurement_energy  = measurement_energy.to(torch.float32) - lnz
        posterior_energy = hef_filter.update(eta_bel_x_t_bar, measurement_energy)
        posterior_energy = posterior_energy.to(torch.float32)
        # Calculate the metrics here 
        mean_nll += fft.neg_log_likelihood(posterior_energy, inputs[:, i])
        # mean_nll_measurement += fft.neg_log_likelihood(measurement_energy, inputs[:, i])
        posterior_pdf = torch.exp(posterior_energy)
        trajectory_list.append(compute_weighted_mean(posterior_pdf,poses,X,Y,T))

        # Visualise the results here
        if torch.isnan(posterior_energy).any() or torch.isnan(mean_nll).any():
            pdb.set_trace()

print("mean nll posterior", mean_nll/TRAJECTORY_LENGTH)
# print("mean nll measurements", mean_nll_measurement/TRAJECTORY_LENGTH)
print("rmse", rmse_se2(torch.stack(trajectory_list), inputs[0]))