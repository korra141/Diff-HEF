import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import sys
import math
import datetime
import random
import pdb
import psutil
import wandb
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)
pid = os.getpid()

torch.cuda.empty_cache()
from src.utils.sampler import se2_grid_samples_torch
from src.distributions.SE2.se2_distribution import SE2, SE2Gaussian
from src.filter.bayes_filter import BayesFilter
from src.filter.Diff_PF import DifferentiablePF
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean, mse
from src.utils.visualisation import plot_se2_mean_filters,plot_se2_filters
import argparse
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
legend = [rf"Predicted belief", rf"Measurement", rf"Posterior"]
CONFIG_MEAN_SE2_LF = [
    {'label': 'HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
        'alpha': 0.8},
    {'label': 'GT', 'c': '#e377c2', 'marker': '*', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
        'zorder': 4, 'alpha': 0.8},
    {'label': 'Beacons', 'c': 'dimgrey', 'marker': 'o', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
        'zorder': 2, 'alpha': 0.8}]

import gc
gc.collect()
torch.cuda.empty_cache()
# device = torch.device("cpu")

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    memory_in_mb = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"[{tag}] Memory Usage: {memory_in_mb:.2f} MB")

def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.01)
        #     module.bias.data = module.bias.data.to(torch.float64)
        # module.weight.data = module.weight.data.to(torch.float64)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.01)
        #     module.bias.data = module.bias.data.to(torch.float64)
        # model.weight.data = model.weight.data.to(torch.float64)

class DensityEstimator(nn.Module):
    def __init__(self, input_size):
        super(DensityEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, input_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.softplus(self.fc3(x))
        return x


def range_pf_step(inputs, measurements, control, landmarks, pf_filter, MOTION_NOISE, MEASUREMENT_NOISE):

    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    # measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
    # measurement_cov_ = torch.tile(measurement_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    belief_hat = pf_filter.prediction(control, motion_model_cov_)

    landmarks = landmarks.to(device)
    measurements = measurements.to(device)
    # observations_cov = observations_cov.to(self.device)
    
    # Ensure proper batch dimension for landmarks
    if landmarks.ndim == 2:
        landmarks = landmarks.unsqueeze(0).expand(self.batch_size, -1, -1)  # (B, L, 2)
    
    weights = pf_filter.weights  # (B, N)
    # Convert to log space
    log_weights = torch.log(weights + 1e-8)  # (B, N)
    
    # For each batch and landmark, compute distances between particles and landmark
    num_landmarks = landmarks.shape[1]
    # print(num_landmarks)
    
    for i in range(num_landmarks):
        # Current landmark for all batches: (B, 1, 2)
        landmark = landmarks[:, i:i+1, :]
        
        # Expand landmark for all particles: (B, 1, 2) -> (B, N, 2)
        landmark_expanded = landmark.expand(-1, pf_filter._N, -1)
        
        # Compute distance between particles and landmark: (B, N)
        distance = torch.norm(belief_hat[:, :, :2] - landmark_expanded, dim=2)
        
        # Current observation for all batches: (B,)
        obs_i = measurements[:, i]
        
        # Current std for all batches: (1)
        std_i = MEASUREMENT_NOISE
        
        # Compute log probabilities: (B, N)
        # Using broadcasting to handle batched computation
        log_prob = Normal(obs_i.unsqueeze(1), std_i).log_prob(distance)
        
        # Update weights: (B, N)
        log_weights += log_prob
    
    posterior_mean = pf_filter.update(log_weights)
    nll_posterior = pf_filter.neg_log_likelihood(inputs)
    return posterior_mean, pf_filter.particles , nll_posterior



class MeasurementDistributionNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=512, num_particles=50):
        super(MeasurementDistributionNet, self).__init__()
        
        self.input_dim = input_dim  # Particle belief state has 3 features
        self.hidden_dim = hidden_dim
        self.num_particles = num_particles
        
        # Define layers
        self.fc1 = nn.Linear(self.num_particles * self.input_dim, self.hidden_dim)
        # self.fc2 = nn.Linear(self.num_particles * self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_particles)
        self.fc4 = nn.Linear(self.hidden_dim, self.num_particles)


        self.conv1 = nn.Conv3d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
    def forward(self, particles, observation):
        """
        Forward pass for the particles.
        particles: tensor of shape [B, N, 3] where B is batch size, N is number of particles.
        
        Returns:
        measurement_distribution: tensor of shape [B, N], the measurement distribution for each particle.
        """
        # pdb.set_trace()
        # Reshape particles to [B * N, 3] for batch processing
        B, N, _ = particles.shape
        # particles_flat = particles.reshape(-1, self.num_particles * self.input_dim) # [B, 1, N*3]
        assert observation.shape == (B, 1), "Observation should have shape [B, 1]"
        z = observation.unsqueeze(1)  # Shape [batch_size, 1, 1]
        z = z.expand(-1, self.num_particles, -1)  # Now z has shape [batch_size, N, 1]
        #p_x_padded = self.input_padding(p_x.unsqueeze(1))
        # Concatenate p_x and z along the channel dimension (axis 1)
        input_ = torch.cat([particles, z], dim=-1)  # Shape [batch_size, N, 4]

        input_flat = input_.reshape(B, -1)  # Flatten to [B, N * 4]
         
        # Pass through the network
        x_1 = self.relu(self.fc1(input_flat))  # Apply ReLU after first layer
        x_2 = self.relu(self.fc2(x_1))      # Apply ReLU after second layer
        # x_3 = self.relu(self.fc3(x_2))      # Apply ReLU after third layer

        # x_1 = self.relu(self.conv1(input_))  # Apply ReLU after first layer
        # x_2 = self.relu(self.conv2(x_1))      # Apply ReLU after second layer
        # x_3 = self.relu(self.conv3(x_2))      # Apply ReLU after third layer

        pred_observations = self.fc3(x_2)  # Output layer
        # Reshape back to [B, N]
        pred_observations = pred_observations.reshape(-1, N, 1)

        observations_var = self.fc4(x_2)  # Output layer for variance

        observations_var = observations_var.reshape(-1, N, 1)

        observations_cov = observations_var ** 2  # Variance
        observations_cov = observations_cov.unsqueeze(-1)
        # observations_cov = torch.diag_embed(observations_var, dim1=-1)  # Create diagonal covariance matrix

        # Compute log likelihoods
        log_likelihoods = torch.distributions.MultivariateNormal(
            loc=pred_observations, scale_tril=observations_cov
        ).log_prob(z)

        # Reshape and return
        return log_likelihoods
        
        # return measurement_pdf


def train_range_pf_step(prior, log_weights, poses, range_beacon, model, inputs, measurements, control, pf_filter, MOTION_NOISE):

    with torch.no_grad():
        motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
        motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
        # start_time_prediction = datetime.datetime.now()
        belief_hat = pf_filter.prediction(prior, control, motion_model_cov_) #[B, N, 3]
        # print(f"Time taken for prediction: {datetime.datetime.now() - start_time_prediction}")
    
    # start_time_forward = datetime.datetime.now()
    measurement_log_likelihood = model(belief_hat.to(torch.float32), measurements.to(torch.float32)) #[B, N]
    # print(f"Time taken for forward pass: {datetime.datetime.now() - start_time_forward}")
    # start_time_weights = datetime.datetime.now()
    updated_log_weights = pf_filter.update_weights(log_weights, measurement_log_likelihood)
    # print(f"Time taken for weight update: {datetime.datetime.now() - start_time_weights}")
    # updated_particles, updated_weights = pf_filter.differentiable_resample_windowed(belief_hat, updated_weights)
    # pdb.set_trace()
    updated_particles_resampled, updated_log_weights_resampled = pf_filter._resample(belief_hat, updated_log_weights)
    # updated_particles_resampled = belief_hat 
    # updated_log_weights_resampled = updated_log_weights
    # pf_filter.lightweight_soft_resampling()
    # start_time_mean = datetime.datetime.now()
    posterior_mean = pf_filter.get_state_estimate(updated_particles_resampled, updated_log_weights_resampled)
    # print(f"Time taken for posterior mean calculation: {datetime.datetime.now() - start_time_mean}")
    diff = updated_particles_resampled - inputs.unsqueeze(1)
    # pdb.set_trace()
    # start_time_posterior = datetime.datetime.now()
    nll_posterior = pf_filter.mixture_likelihood(diff,  updated_log_weights_resampled, reduce_mean=True)
    # print(f"Time taken for posterior NLL calculation: {datetime.datetime.now() - start_time_posterior}")
    # start_time_measurement = datetime.datetime.now()
    # nll_measurement = pf_filter.neg_log_likelihood_measurement(poses, range_beacon, measurements, updated_weights)
    # print(f"Time taken for measurement NLL calculation: {datetime.datetime.now() - start_time_measurement}")

    return posterior_mean, updated_particles_resampled, updated_log_weights_resampled, nll_posterior




def validate(model, epoch, val_loader, poses, X, Y, T, diff_fft, hed_r1, device, args):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_nll_posterior = 0
    total_nll_likelihood = 0
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    beacons = torch.tensor(
                [[0, 0.1],
                [0, 0.05],
                [0, 0.0],
                [0, -0.05],
                [0, -0.1]]).to(device)

    with torch.no_grad():
        for data in val_loader:
            inputs, measurements, control, beacon_id = data
            inputs, measurements, control, beacon_id = inputs.to(device), measurements.to(device), control.to(device), beacon_id.to(device)

            cov_prior = torch.diag(torch.tensor(args.cov_prior)).to(device)
            prior_diff = SE2Gaussian(inputs[:, 0], cov_prior, samples=poses, fft=diff_fft)
            prior_diff.normalize()
            trajectory_list = []
            diff_hef_filter = BayesFilter(distribution=SE2, prior=prior_diff, device=device)

            for i in range(args.trajectory_length - 1):
                traj_idx = i + 1
                range_beacon = beacons[beacon_id[:, i], :]
                posterior, measurement_model, belief_hat, predicted_pose, nll_measurement_likelihood, nll_posterior = diff_hef_step(
                    inputs[:, traj_idx], measurements[:, i], range_beacon, control[:, i], poses, X, Y, T, args.grid_size, diff_hef_filter, model, diff_fft, hed_r1, MOTION_NOISE, args.batch_size
                )

                # loss = (nll_measurement_likelihood + nll_posterior).to(torch.float32)
                # total_loss += float(loss.item())
                trajectory_list.append(predicted_pose)
                total_nll_likelihood += float(nll_measurement_likelihood.item())
                total_nll_posterior += float(nll_posterior.item())

            predicted_trajectory = torch.stack(trajectory_list, dim=1)
            total_rmse += rmse_se2(inputs, predicted_trajectory)

    len_trajectory_adjusted = args.trajectory_length - 1
    avg_rmse = total_rmse / len(val_loader)
    avg_nll_likelihood = total_nll_likelihood / (len(val_loader) * len_trajectory_adjusted)
    avg_nll_posterior = total_nll_posterior / (len(val_loader) * len_trajectory_adjusted)

    wandb.log({
        "Validation RMSE": avg_rmse,
        "Validation NLL Likelihood": avg_nll_likelihood,
        "Validation NLL Posterior": avg_nll_posterior,
        "Epoch": epoch,
    })

    print(f"RMSE: {avg_rmse:.4f}, NLL Likelihood: {avg_nll_likelihood:.4f}, NLL Posterior: {avg_nll_posterior:.4f}")
    return avg_rmse, avg_nll_likelihood, avg_nll_posterior

def main(args, logging_path):
    # parameters
    NUM_EPOCHS = args.num_epochs
    NUM_TRAJECTORIES = args.num_trajectories
    TRAJECTORY_LENGTH = args.trajectory_length
    STEP_MOTION = SE2Group(args.step_motion[0], args.step_motion[1], args.step_motion[2])
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    MEASUREMENT_NOISE = np.sqrt(args.measurement_cov)
    batch_size = args.batch_size
    validation_split = args.validation_split
    grid_size = tuple(args.grid_size)
    cov_prior = torch.diag(torch.tensor(args.cov_prior)).to(torch.float64).to(device)
    cov_prior_batch = torch.tile(cov_prior.unsqueeze(0), [args.batch_size, 1, 1])
    range_x = (-0.5, 0.5)
    range_y = (-0.5, 0.5)
    beacons = torch.tensor(
                [[0, 0.1],
                [0, 0.05],
                [0, 0.0],
                [0, -0.05],
                [0, -0.1]]).to(device)
    poses, X, Y, T = se2_grid_samples_torch(batch_size, grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    decay_rate = args.decay_rate 
    # Generate dataset
    train_loader, val_loader, _ = generate_bounded_se2_dataset(
        num_trajectories=NUM_TRAJECTORIES,
        trajectory_length=TRAJECTORY_LENGTH,
        step_motion=STEP_MOTION,
        motion_noise=MOTION_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
        samples=poses.cpu().numpy(),
        batch_size=batch_size, 
        validation_split=validation_split,
        test_split=args.test_split
    )
    # model = DensityEstimator(math.prod(grid_size)).to(device)
    model = MeasurementDistributionNet(input_dim=4, hidden_dim=1024, num_particles=math.prod(grid_size)).to(device)
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    lr_decay = lambda epoch : (args.learning_rate_end / args.learning_rate_start) ** ((epoch / NUM_EPOCHS) ** args.slope_weight)
    torch.autograd.set_detect_anomaly(True)
    diff_pf_filter = DifferentiablePF(math.prod(grid_size), batch_size, grid_size, device=device, soft_resample_alpha=args.soft_resample_alpha)
    for epoch in range(NUM_EPOCHS):
        loss_tot = 0
        mean_rmse_tot = 0
        nll_posterior_tot = 0
        nll_likelihood_tot = 0

        start_time = datetime.datetime.now()
         # Adjust this value to control the decay speed
        regularizer_weight = math.exp(-decay_rate * ((epoch - args.threshold_warmup) / (NUM_EPOCHS- args.threshold_warmup)))
        sample_batch = random.randint(0, len(train_loader) - 1)
        new_lr = args.learning_rate_start * lr_decay(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # validate(model, epoch, val_loader, poses, X, Y, T, diff_fft, hed_r1, device, args)
        for j, data in enumerate(train_loader):
            # print_memory_usage(f"Batch {j} Epoch {epoch}")
            start_time_step = datetime.datetime.now()
            inputs, measurements, control, beacon_id = data
            inputs, measurements, control, beacon_id = inputs.to(device), measurements.to(device), control.to(device), beacon_id.to(device)
            # Generate particles from prior: (B, N, 3)
            L = torch.linalg.cholesky(cov_prior_batch)  # (B, 3, 3)
            noise = torch.randn((batch_size, 3, math.prod(grid_size)), device=device, dtype=torch.float64)  # (B, 3, N)
            # L @ noise -> (B, 3, N), then transpose and add prior
            particles_noise = torch.bmm(L, noise)  # (B, 3, N)
            prior_pf = particles_noise.permute(0, 2, 1) + inputs[:, 0].unsqueeze(1)  # (B, N, 3)
            
            log_weight_pf = torch.ones((batch_size, math.prod(grid_size)), device=device, dtype=torch.float64) * (-np.log(math.prod(grid_size)))  # (B, N)
            trajectory_list = []
            trajectory_list.append(inputs[:, 0])  # Append the initial pose
            sample_idx = random.randint(0, batch_size - 1)
            # Perform operations on inputs and labels using HEF analytical filter here
            for i in range(TRAJECTORY_LENGTH - 1):
                traj_idx = i + 1
                start_time_step = datetime.datetime.now()

                
                range_beacon = beacons[beacon_id[:, i], :]

                # start_time_step = datetime.datetime.now()
                posterior_mean_pf, posterior_pf, updated_log_weights, nll_posterior_pf = train_range_pf_step(prior_pf,  log_weight_pf, poses, range_beacon, model, inputs[:, traj_idx], measurements[:, i], control[:, i], diff_pf_filter, MOTION_NOISE=MOTION_NOISE)
                # print(f"Time taken for differentiable PF step: {datetime.datetime.now() - start_time_step}")
                log_weight_pf = updated_log_weights.detach()
                prior_pf = posterior_pf.detach()
                # pdb.set_trace()
                # if epoch < args.threshold_warmup:
                # loss = nll_posterior_pf.to(torch.float32)
                loss = 0.5*(nll_posterior_pf.to(torch.float32) + mse(posterior_mean_pf, inputs[:, traj_idx]).to(torch.float32))
                # else:
                #     loss = (regularizer_weight * nll_measurement_pf + (1 - regularizer_weight) * nll_posterior_pf).to(torch.float32)
                # loss = (regularizer_weight * nll_measurement_likelihood + (1-regularizer_weight) * nll_posterior).to(torch.float32)
                optimizer.zero_grad()
                loss.backward()
                # loss.backward(retain_graph=True)
                optimizer.step()
                with torch.no_grad():
                    trajectory_list.append(posterior_mean_pf)
                    loss_tot += float(loss.item())
                    # nll_likelihood_tot += float(nll_measurement_pf.item())
                    nll_posterior_tot += float(nll_posterior_pf.item())

                # print(f"Processing Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {j + 1}/{len(train_loader)}, Step {i + 1}/{TRAJECTORY_LENGTH - 1}, Time: {datetime.datetime.now() - start_time_step}")

                # Visualise ther results 
            predicted_trajectory = torch.stack(trajectory_list, dim=1)

            with torch.no_grad():
                mean_rmse_tot += rmse_se2(inputs, predicted_trajectory)
                # mean_rmse_true_tot += rmse_se2(inputs, predicted_trajectory_true)

            # print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {j + 1}/{len(train_loader)}, Time: {datetime.datetime.now() - start_time_step}")

        # Create a table for logging
        end_time = datetime.datetime.now()
        len_trajectory_length_adjusted = TRAJECTORY_LENGTH - 1
        table_data = [
            ["Epoch", epoch],
            ["Time", str(end_time - start_time)],
            ["Loss", loss_tot / (len_trajectory_length_adjusted * len(train_loader))],
            # ["NLL Likelihood", nll_likelihood_tot / (len_trajectory_length_adjusted * len(train_loader))],
            ["NLL Posterior", nll_posterior_tot / (len_trajectory_length_adjusted * len(train_loader))],
            ["RMSE", mean_rmse_tot / len(train_loader)],
        ]
        wandb.log({ 
                "Epoch": epoch,
                "Loss": loss_tot / (len_trajectory_length_adjusted * len(train_loader)),
                # "NLL Likelihood": nll_likelihood_tot / (len_trajectory_length_adjusted * len(train_loader)), 
                "NLL Posterior": nll_posterior_tot / (len_trajectory_length_adjusted * len(train_loader)), 
                "RMSE": mean_rmse_tot / len(train_loader),
                "Regularizer Weight": regularizer_weight,
                "Learning Rate": new_lr,
            })

        # Print the table
        print("\n" + "-" * 40)
        for row in table_data:
            print(f"{row[0]:<30} {row[1]:<10}")
        print("-" * 40)
        # Save the model
        if (epoch + 1) % 20 == 0:
            model_save_path = os.path.join(logging_path, f"measurement_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path, base_path=logging_path)


def visualize_trajectory(pose_dict,posterior_pdf, measurement_pdf, belief_hat_prob, beacons, sample_index, traj_idx, beacon_idx, X, Y, T, legend, CONFIG_MEAN_SE2_LF, logging_path, name):
    axes_mean = plot_se2_mean_filters(
        [belief_hat_prob[sample_index], measurement_pdf[sample_index], posterior_pdf[sample_index]],
        X, Y, T,
        samples=pose_dict, iteration=traj_idx, beacons=beacons[:, :2],
        level_contours=False, contour_titles=legend, config=CONFIG_MEAN_SE2_LF)

    for ax in axes_mean:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    axes_mean[3].scatter(beacons[beacon_idx[sample_index, traj_idx], 0].cpu(), beacons[beacon_idx[sample_index, traj_idx], 1].cpu(),
                            c='y', marker='o', s=80, alpha=0.8, zorder=2)
                            
    plt.savefig(logging_path + f"/se2_test_landmarks{name}.png")
    plt.close()


import ast

def parse_list(value):
    """Helper function to parse a list from string input."""
    try:
        return ast.literal_eval(value)  # Safely convert the string into a Python literal (e.g., list)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_trajectories', type=int, default=300, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=80, help='Length of each trajectory')
    parser.add_argument('--step_motion', type=parse_list, default=[0.01, 0.00, np.pi / 40], help='Step motion parameters')
    parser.add_argument('--motion_cov', type=parse_list, default=[0.001, 0.001, 0.001], help='Motion noise parameters')
    parser.add_argument('--measurement_cov', type=float, default=0.0001, help='Measurement noise')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.12, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--grid_size', type=parse_list, default=[50, 50, 32], help='Grid size')
    parser.add_argument('--cov_prior', type=parse_list, default=[0.1, 0.1, 0.1], help='Covariance prior')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--decay_rate', type=float, default=5, help='Decay rate for regularization')
    parser.add_argument('--threshold_warmup', type=int, default=100, help='Threshold for warmup')
    parser.add_argument('--learning_rate_start', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Final learning rate')
    parser.add_argument('--slope_weight', type=float, default=0.5, help='Slope weight for learning rate decay')
    parser.add_argument('--soft_resample_alpha', type=float, default=0.75, help='Soft resampling alpha for DifferentiablePF')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    run = wandb.init(project="Diff-PF",group="SE2",entity="korra141",
              tags=["SE2","Training", "NLP+MSE", "Resampling"],
              name="SE2-DiffPF-RangeSimulator-1",
              notes="Diff-PF on SE2 Range Simulator",
              config=args)
    # run = wandb.init(mode="disabled", project="Diff-EKF",group="SE2",entity="korra141",
    #           tags=["SE2","DenistyEstimation","UnimodalNoise","Training"],
    #           name="SE2-DiffEKF-RangeSimulator-1",
    #           notes="Diff-EKF on SE2 Range Simulator",
    #           config=args)
    # artifact = wandb.Artifact("SE2_Range_DiffHEF", type="script")
    # artifact.add_file(__file__)
    # run.log_artifact(artifact)
    run_name = "SE2_Range_DiffPF"
    # run_name = "SE2_Range_EKFHEF"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
    main(args,logging_path)
    # training_ekf(logging_path, args)
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_DiffHEF/20250314_163823_7387/measurement_model_epoch_50.pth"
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_DiffHEF/20250316_231533_3842/measurement_model_epoch_200.pth"
    # inference_old_hef(model_path, args, logging_path)
    # inference(args, logging_path, diff_model_path=None)