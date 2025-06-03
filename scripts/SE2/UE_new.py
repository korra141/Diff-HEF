import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import sys
import math
import datetime
import random
import torch.multiprocessing as mp
import logging
import uuid
import time
import torch.nn.functional as F
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
from src.data_generation.SE2.uncertainity_estimation import create_dataloaders, SE2Group

train_loader, val_loader = create_dataloaders(NUM_TRAJECTORIES, TRAJECTORY_LENGTH, STEP_MOTION, MOTION_NOISE, MEASUREMENT_NOISE, batch_size, validation_split) 

# class MLP(nn.Module):
#     def __init__(self, input_dim=3, output_dim=(50, 50, 32), hidden_dims=[128, 256, 512]):
#         super(MLP, self).__init__()
#         self.output_dim = output_dim
#         layers = []
#         current_dim = input_dim
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(current_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             current_dim = hidden_dim
        
#         final_output_dim = torch.prod(torch.tensor(output_dim)).item() 
#         layers.append(nn.Linear(current_dim, final_output_dim))
        
#         self.mlp = nn.Sequential(*layers)
    
#     def forward(self, x):
#         x = self.mlp(x)
#         x = nn.ReLU()(x)
#         x = x.view(x.size(0), *self.output_dim)
#         return x



# class UpsampleDensityEstimator(nn.Module):
#     def __init__(self, input_dim=3, output_shape=(50, 50, 32)):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_shape = output_shape
        
#         # Compute intermediate grid sizes for convolution
#         intermediate_dim = (output_shape[0] // 4, output_shape[1] // 4)  # Example: downsample by factor of 4
        
#         # Fully connected layers for initial latent representation
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, intermediate_dim[0] * intermediate_dim[1] * 16),
#             nn.ReLU(),
#         )
        
#         # Deconvolution layers for upsampling to final shape
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, output_shape[2], kernel_size=3, padding=1),
#             nn.Sigmoid()  # Optional for density maps in [0, 1]
#         )
        
#     def forward(self, x):
#         # Fully connected to latent grid representation
#         latent = self.fc(x)  # (N, input_dim) -> (N, intermediate_dim[0] * intermediate_dim[1] * 16)
#         latent = latent.view(-1, 16, self.output_shape[0] // 4, self.output_shape[1] // 4)  # Reshape
        
#         # Deconvolution for upsampling
#         output = self.deconv(latent).permute(0, 2, 3, 1)  # (N, output_shape[2], output_shape[0], output_shape[1])
#         return output

# class ConvDensityEstimator(nn.Module):
#     def __init__(self, input_shape=(50, 50, 32)):
#         super(ConvDensityEstimator, self).__init__()
#         self.input_shape = input_shape
        
#         self.conv_layers = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
#             nn.ReLU()  # Optional for density maps in [0, 1]
#         )
        
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         density = self.conv_layers(x)
#         return density.squeeze(1)

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

class R2DensityEstimator(nn.Module):
    def __init__(self, grid_size=(50, 50)):
        super(R2DensityEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.input_padding = nn.ReplicationPad2d(1)
        
    def forward(self, r2_input):
        r2_input = self.input_padding(r2_input)
        r2_density = self.network(r2_input).squeeze(1)  # Shape: [batch_size, H, W]
        return r2_density

class S1DensityEstimator(nn.Module):
    def __init__(self, grid_size=32):
        super(S1DensityEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1, 10),  # Input: theta (1D)
            nn.ReLU(),
            nn.Linear(10, grid_size),  # Output: density on S1
            nn.ReLU()  # Ensure non-negative density
        )
        
    def forward(self, s1_input):
        s1_density = self.network(s1_input)  # Shape: [batch_size, grid_size]
        return s1_density

# class IndependentDensityEstimator(nn.Module):
#     def __init__(self, grid_size=(50, 50, 32)):
#         super(IndependentDensityEstimator, self).__init__()
        
#         self.r2_estimator = R2DensityEstimator(grid_size=grid_size[:2])
#         self.s1_estimator = S1DensityEstimator(grid_size=grid_size[2])
        
#     def forward(self, r2_input, s1_input):
#         r2_density = self.r2_estimator(r2_input)
#         s1_density = self.s1_estimator(s1_input)
        
#         combined_density = r2_density.unsqueeze(-1) * s1_density.unsqueeze(1).unsqueeze(1)  # Independent product
#         return combined_density, r2_density, s1_density



num_epochs = 1000
input_dim = 3
grid_size = (50, 50, 32)
learning_rate = 1e-3
lr_factor = 10
learning_rate_end = 1e-5

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
logging.info(f"Initial Covariance: {INITIAL_COV}")
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
# model = IndependentDensityEstimator(grid_size).to(device)

model_r2 = R2DensityEstimator(grid_size[:2]).to(device)
model_s1 = S1DensityEstimator(grid_size[2]).to(device)

initialize_weights(model_r2)
initialize_weights(model_s1)

model_r2.train()  # Set the model to training mode
model_s1.train()  # Set the model to training mode

optimizer_r2 = optim.Adam(model_r2.parameters(), lr=learning_rate)
optimizer_s1 = optim.Adam(model_s1.parameters(), lr=learning_rate)

# def forward_with_checkpoint(model, *inputs):
#     return checkpoint(model, *inputs)

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
        
        start_time_batch = time.time()
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        initial_distribution_r2 = GaussianDistribution_r2(inputs[:,0:2], true_cov_r2.to(device),grid_size[0:2],x_range=[-0.5,0.5],y_range=[-0.5,0.5])
        initial_density_r2 = initial_distribution_r2.density_over_grid().to(torch.float32)

        torch.cuda.synchronize()

        start_time_forward = time.time()

        output_r2 = model_r2(initial_density_r2.unsqueeze(1))
        output_s1 = model_s1(inputs[:,2].unsqueeze(1))

        torch.cuda.synchronize()

        print(f"Time taken for forward pass {time.time() - start_time_forward:.2f}s")

        start_time_fft = time.time()
        output_se2 =
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
        # do not 
        loss = nll 
        # + scaling_factor * smoothness_loss
        # Backward pass
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        optimizer.zero_grad()
        loss.backward()
        # total_norm = 0.0
        # # for param in model.parameters():
        # #     if param.grad is not None:  # Check if the gradient is computed
        # #         param_norm = param.grad.data.norm(2)  # L2 norm of the gradient
        # #         total_norm += param_norm.item() ** 2

        # total_norm = total_norm ** 0.5  # Final L2 norm
        # print("Gradient Norm (L2):", total_norm)
        # if (batch_idx + 1) % accumulation_steps == 0:
            # scaler.step(optimizer)
            # scaler.update()
        
        optimizer.step()
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
    



