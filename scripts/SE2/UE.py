import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import sys
import math
import datetime
import random
base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)

import pdb
torch.cuda.empty_cache()
from src.utils.visualisation import plot_density, plot_distributions_s1
from src.distributions.R2.StandardDistribution import GaussianDistribution as GaussianDistribution_r2
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution,VonMissesDistribution_torch
from src.distributions.SE2.SE2_FFT import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from torch.utils.checkpoint import checkpoint





class SE2Group:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta
  def __add__(self, other):
    x = self.x + other.x * np.cos(self.theta) - other.y * np.sin(self.theta)
    y = self.y + other.y * np.cos(self.theta) + other.x * np.sin(self.theta)
    theta = self.theta + other.theta
    return SE2Group(x, y, theta)
  def parameters(self):
    return np.array([self.x, self.y, self.theta])
  @classmethod
  def from_parameters(cls, x, y, theta):
    return cls(x, y, theta)

class SE2SimpleSimulator:

  def __init__(self, start, step, measurement_noise, motion_noise):
    self.position = start
    self.step = step
    self.motion_noise = motion_noise
    self.measurement_noise = measurement_noise
    self.beacons = np.array(
            [[0, 0.1],
             [0, 0.05],
             [0, 0.0],
             [0, -0.05],
             [0, -0.1]])
    self.beacon_idx = 0

  def motion(self):

    self.position = self.position + self.step
    noisy_prediction = self.step.parameters() + np.random.randn(3) * self.motion_noise
    noisy_prediction[2] = (noisy_prediction[2] + np.pi) % (2*np.pi) - np.pi
    self.position.theta = (self.position.theta + np.pi) % (2*np.pi) - np.pi
    positions = self.position.parameters()
    positions [0:2] = (positions[0:2] + 0.5) % 1.0 - 0.5
    return positions  , noisy_prediction
#   def measurement(self):

#     self._update_beacon_idx()
#     range_beacon = self.beacons[self.beacon_idx, :]
#     # Observation z_t
#     self.range_measurement = np.linalg.norm(self.position.parameters()[0:2] - range_beacon)
#     # Jitter range measurement with noise
#     self.range_measurement += np.random.normal(0.0, self.measurement_noise, 1).item()

#     return self.range_measurement

  def measurement(self):
    measurement_ = self.position.parameters() + np.random.randn(3) * self.measurement_noise
    measurement_[0:2] = (measurement_[0:2] + 0.5) % 1.0 - 0.5
    measurement_[2] = (measurement_[2]  + np.pi ) % (2*np.pi) - np.pi
    return measurement_

  def _update_beacon_idx(self) -> None:
    """
    Update beacon index, and cycle back to 0 if need be.
    """
    self.beacon_idx += 1
    if self.beacon_idx >= self.beacons.shape[0]:
        self.beacon_idx = 0

def random_start_pose():
    """
    Generate a random start pose within the specified bounds.
    :return: A random pose [x, y, theta] in [-0.5, 0.5] for x, y and [0, 2pi] for theta.
    """
    x = np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    theta = np.random.uniform(-np.pi, np.pi)
    return SE2Group(x, y, theta)

def generate_bounded_se2_dataset(
    num_trajectories,
    trajectory_length,
    step_motion,
    motion_noise,
    measurement_noise,
):
    """
    Generates a dataset of SE2 trajectories and corresponding measurements within bounded space.

    :param num_trajectories: Number of trajectories to generate.
    :param trajectory_length: Number of steps per trajectory.
    :param step_motion: Step motion [dx, dy, dtheta].
    :param motion_noise: Noise for motion [x, y, theta].
    :param measurement_noise: Noise for measurements [x, y, theta].
    :param output_file: File to save the generated dataset.
    """
    true_trajectories = np.ndarray((num_trajectories, trajectory_length, 3))
    measurements = np.ndarray((num_trajectories, trajectory_length,3))
    noisy_control = np.ndarray((num_trajectories, trajectory_length, 3))


    for traj_id in range(num_trajectories):
        # Initialize simulator with a random start pose
        start_pose = random_start_pose()
        simulator = SE2SimpleSimulator(
            start=start_pose,
            step=step_motion,
            measurement_noise=measurement_noise,
            motion_noise=motion_noise,
        )

        for step in range(trajectory_length):
            # Simulate motion
            motion, noisy_step = simulator.motion()
            true_trajectories[traj_id, step, :] = motion
            noisy_control[traj_id, step, :] = noisy_step

            # Simulate measurement
            measurements_ = simulator.measurement()
            measurements[traj_id, step] = measurements_


    return true_trajectories, noisy_control, measurements

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

NUM_TRAJECTORIES = 100
TRAJECTORY_LENGTH = 10
STEP_MOTION = SE2Group(0.05, 0.05, np.pi / 20)
MOTION_NOISE = np.array([0.01, 0.01, 0.005])
MEASUREMENT_NOISE = np.array([0.2, 0.2, 0.2])
INITIAL_COV = np.array([0.01, 0.01, 0.01])
batch_size = 20
validation_split = 0.2

# Generate dataset
true_trajectories, noisy_control, measurements = generate_bounded_se2_dataset(
    num_trajectories=NUM_TRAJECTORIES,
    trajectory_length=TRAJECTORY_LENGTH,
    step_motion=STEP_MOTION,
    motion_noise=MOTION_NOISE,
    measurement_noise=MEASUREMENT_NOISE,
)

measurements_torch = torch.from_numpy(measurements).type(torch.FloatTensor)
ground_truth_torch = torch.from_numpy(true_trajectories).type(torch.FloatTensor)
ground_truth_torch = torch.flatten(ground_truth_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
measurements_torch = torch.flatten(measurements_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(ground_truth_torch, measurements_torch)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, shuffle=False,  pin_memory=True)
# val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, shuffle=False,  pin_memory=True)
        
class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=(50, 50, 32), hidden_dims=[128, 256, 512]):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        final_output_dim = torch.prod(torch.tensor(output_dim)).item() 
        layers.append(nn.Linear(current_dim, final_output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.mlp(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), *self.output_dim)
        return x

import torch
import logging
import uuid
import time
import torch.nn as nn

class UpsampleDensityEstimator(nn.Module):
    def __init__(self, input_dim=3, output_shape=(50, 50, 32)):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        
        # Compute intermediate grid sizes for convolution
        intermediate_dim = (output_shape[0] // 4, output_shape[1] // 4)  # Example: downsample by factor of 4
        
        # Fully connected layers for initial latent representation
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, intermediate_dim[0] * intermediate_dim[1] * 16),
            nn.ReLU(),
        )
        
        # Deconvolution layers for upsampling to final shape
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[2], kernel_size=3, padding=1),
            nn.Sigmoid()  # Optional for density maps in [0, 1]
        )
        
    def forward(self, x):
        # Fully connected to latent grid representation
        latent = self.fc(x)  # (N, input_dim) -> (N, intermediate_dim[0] * intermediate_dim[1] * 16)
        latent = latent.view(-1, 16, self.output_shape[0] // 4, self.output_shape[1] // 4)  # Reshape
        
        # Deconvolution for upsampling
        output = self.deconv(latent).permute(0, 2, 3, 1)  # (N, output_shape[2], output_shape[0], output_shape[1])
        return output

class ConvDensityEstimator(nn.Module):
    def __init__(self, input_shape=(50, 50, 32)):
        super(ConvDensityEstimator, self).__init__()
        self.input_shape = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()  # Optional for density maps in [0, 1]
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        density = self.conv_layers(x)
        return density.squeeze(1)

def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def init_weights_mlp(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


import torch
import torch.nn as nn
import torch.nn.functional as F

class IndependentDensityEstimator(nn.Module):
    def __init__(self, grid_size=(50, 50, 32)):
        super(IndependentDensityEstimator, self).__init__()
        
        # Sub-network for R2
        self.r2_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Sub-network for S1
        self.s1_network = nn.Sequential(
            nn.Linear(1, 10),  # Input: theta (1D)
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(10, grid_size[2]),  # Output: density on S1
            nn.ReLU()  # Ensure non-negative density
        )
        
    def forward(self, r2_input, s1_input):
        # Estimate density on R2
        r2_density = self.r2_network(r2_input).squeeze(1)  # Shape: [batch_size, H, W]
        # r2_density = torch.mean(r2_density, dim=(1, 2))  # Aggregate spatial dimensions
        
        # Estimate density on S1
        s1_density = self.s1_network(s1_input) # Shape: [batch_size, B]
        
        # Combine densities
        combined_density = r2_density.unsqueeze(-1) * s1_density.unsqueeze(1).unsqueeze(1)  # Independent product
        return combined_density, r2_density, s1_density



num_epochs = 1000
input_dim = 3
grid_size = (50, 50, 32)
learning_rate = 1e-3
lr_factor = 5
learning_rate_end = 1e-6

run_name = "UE_SE2"
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
random_number = random.randint(1000, 9999)
logging_path = os.path.join(base_path,"logs", run_name, current_datetime + "_" + str(random_number))
os.makedirs(logging_path, exist_ok=True)

# Set up logging
log_file = os.path.join(logging_path, "training.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log initial information
logging.info("Starting training")
logging.info(f"Number of trajectories: {NUM_TRAJECTORIES}")
logging.info(f"Trajectory length: {TRAJECTORY_LENGTH}")
logging.info(f"Step motion: {STEP_MOTION.parameters()}")
logging.info(f"Motion noise: {MOTION_NOISE}")
logging.info(f"Measurement noise: {MEASUREMENT_NOISE}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Validation split: {validation_split}")
logging.info(f"Number of epochs: {num_epochs}")
logging.info(f"Learning rate: {learning_rate}")
logging.info(f"Learning rate end: {learning_rate_end}")
logging.info(f"Learning rate decay factor: {lr_factor}")


# model = MLP(input_dim=3, output_dim=grid_size).to(device)
# model = UpsampleDensityEstimator(input_dim=3, output_shape=grid_size).to(device)
# model = ConvDensityEstimator(input_shape=grid_size).to(device)
model = IndependentDensityEstimator(grid_size).to(device)
initialize_weights(model)
model.train()  # Set the model to training mode
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def forward_with_checkpoint(model, *inputs):
    return checkpoint(model, *inputs)

fft = SE2_FFT(spatial_grid_size=grid_size,
                  interpolation_method='spline',
                  spline_order=1,
                  oversampling_factor=1)
true_cov_r2 = torch.tile(torch.tensor([MEASUREMENT_NOISE[0]**2,MEASUREMENT_NOISE[1]**2]).unsqueeze(0),(batch_size,1))
true_cov_se2 = torch.tensor([MEASUREMENT_NOISE[0]**2,MEASUREMENT_NOISE[1]**2,MEASUREMENT_NOISE[2]**2])
inital_cov_se2 = torch.diag(torch.tensor(INITIAL_COV)).to(device).to(torch.float32)
initial_cov_r2 = torch.tile(torch.tensor([INITIAL_COV[0],INITIAL_COV[1]]).unsqueeze(0),(batch_size,1))

poses, X, Y, T = se2_grid_samples_torch(batch_size ,grid_size)
poses = poses.to(device)
scaling_factor = 1
sample_batch = 0
accumulation_steps = 5
scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    running_loss = 0.0
    true_nll_input_tot = 0.0
    nll_tot = 0.0
    smoothness_tot = 0.0
    learning_rate_decay = learning_rate + (learning_rate_end - learning_rate) * (1 - np.exp((-lr_factor * epoch / num_epochs)))
    logging.info(f"learning_rate {learning_rate_decay}")
    start_time = time.time()
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_decay
    # print(f"Time take for learning rate decay {time.time() - start_time:.2f}s")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        with torch.cuda.amp.autocast():
            start_time_batch = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            initial_density = GaussianDistribution_se2(inputs.unsqueeze(1), inital_cov_se2 ,grid_size).density(poses)
            initial_density = initial_density.reshape((batch_size, *grid_size))
            initial_distribution_r2 = GaussianDistribution_r2(inputs[:,0:2], true_cov_r2.to(device),grid_size[0:2],x_range=[-0.5,0.5],y_range=[-0.5,0.5])
            initial_density_r2 = initial_distribution_r2.density_over_grid().to(torch.float32)
            # print(initial_density_r2.shape)
            initial_distribution_s1 = VonMissesDistribution_torch(inputs[:,2].unsqueeze(-1), torch.tensor(INITIAL_COV[2]), grid_size[2])
            initial_density_s1  = initial_distribution_s1.density()
            # print(initial_density_s1.shape)

            # Forward pass
            # output = model(initial_density)
            torch.cuda.synchronize()
            start_time_forward = time.time()
            initial_density_r2.requires_grad = True
            inputs[:,2].requires_grad = True
            output, r2_density, s1_density = forward_with_checkpoint(model, initial_density_r2.unsqueeze(1), inputs[:,2].unsqueeze(1))
            # print(output)
            torch.cuda.synchronize()
            # print(f"Time taken for forward pass {time.time() - start_time_forward:.2f}s")
            # print(output.shape)
            start_time_fft = time.time()
            _, z = fft.compute_moments_lnz(torch.log(output + 1e-8))
            
            output = output.to(device)
            z = z.to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            output_density = output/z
            # Compute loss
            nll = torch.mean(fft.neg_log_likelihood(torch.log(output_density + 1e-8), targets))
            # print(f"Time taken for fft {time.time() - start_time_fft:.2f}s")
        # diff_dim1 = output_density[1:, :, :] - output_density[:-1, :, :]  # Differences along the first dimension
        # diff_dim2 = output_density[:, 1:, :] - output_density[:, :-1, :]  # Differences along the second dimension
        # diff_dim3 = output_density[:, :, 1:] - output_density[:, :, :-1]  # Differences along the third dimension

        # # Smoothness loss: squared differences summed over all dimensions
        # smoothness_loss = (
        #     torch.mean(diff_dim1 ** 2) +
        #     torch.mean(diff_dim2 ** 2) +
        #     torch.mean(diff_dim3 ** 2)
        # )
            torch.cuda.synchronize()
            start_time_backward = time.time()
            loss = nll 
        # + scaling_factor * smoothness_loss
        # Backward pass
        # optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        # total_norm = 0.0
        # # for param in model.parameters():
        # #     if param.grad is not None:  # Check if the gradient is computed
        # #         param_norm = param.grad.data.norm(2)  # L2 norm of the gradient
        # #         total_norm += param_norm.item() ** 2

        # total_norm = total_norm ** 0.5  # Final L2 norm
        # print("Gradient Norm (L2):", total_norm)
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
        # optimizer.zero_grad()
        # optimizer.step()
        torch.cuda.synchronize()
        # print(f"Time taken for backward pass {time.time() - start_time_backward:.2f}s")
        target_distribution_r2 = GaussianDistribution_r2(inputs[:,0:2], true_cov_r2.to(device),grid_size[:-1],x_range=[-0.5,0.5],y_range=[-0.5,0.5])
        true_density_plot_r2 = target_distribution_r2.density_over_grid()

        
        cov = torch.diag(true_cov_se2.to(device)).to(torch.float32)
        energy_se2  = GaussianDistribution_se2(inputs.unsqueeze(1), cov,grid_size).energy(poses)
        energy_se2 = energy_se2.reshape((batch_size, *grid_size))
        nll_se2 = fft.neg_log_likelihood(energy_se2, targets)

        running_loss += loss.item()
        nll_tot += nll.item()
        # smoothness_tot += smoothness_loss.item()
        true_nll_input_tot += nll_se2.mean().item()
        if epoch % 10 == 0 and batch_idx == sample_batch:
            indices = np.random.choice(batch_size, 3, replace=False)
            for j in indices:
                plot_dict = {}
                plot_dict = {
                    'true_density': true_density_plot_r2[j],
                    'predicted_density': r2_density[j],
                    # torch.sum(output_density[j],dim=2) * (2*math.pi/grid_size[2]),
                    }
                plot_density(inputs[j,0:2], targets[j,0:2],[-0.5,0.5],[-0.5,0.5],logging_path,plot_dict,f"training_r2_epoch_{epoch}_batch_{batch_idx}_sample{j}")
                
                # predicted_density = torch.sum(output_density[j],dim=(0,1)) * (1/grid_size[0]) * (1/grid_size[1])
                plot_distributions_s1(MEASUREMENT_NOISE[2], grid_size[2], torch.log(s1_density[j] + 1e-10), inputs[:,2] + math.pi, targets[:,2] + math.pi, epoch, batch_idx, j, logging_path)
        # print(f"Time taken for batch {time.time() - start_time_batch:.2f}s")
    epoch_loss = running_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Estimated NLL: {nll_tot/ len(train_loader):.4f}, Smoothening Loss: {smoothness_tot/len(train_loader):.4f}, True NLL: {true_nll_input_tot/len(train_loader):.4f}, Time: {time.time() - start_time:.2f}s")
    



