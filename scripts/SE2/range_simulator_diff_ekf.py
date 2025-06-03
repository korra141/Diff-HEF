import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.utils.sampler import se2_grid_samples_torch
from src.distributions.SE2.se2_distribution import SE2, SE2Gaussian
from src.filter.bayes_filter import BayesFilter
from src.filter.EKF import RangeEKF
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean,mse
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

def range_ekf_step(inputs, measurements, control, range_beacon, ekf_filter, pose, MOTION_NOISE, MEASUREMENT_NOISE):
    start_time = datetime.datetime.now()
    # Prediction step
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
    batch_size = inputs.shape[0]
    measurement_cov_ = torch.tile(measurement_cov.unsqueeze(0), [batch_size , 1,  1])
    _, _ = ekf_filter.prediction(control, motion_model_cov_)
    q = landmarks - pose[:, :, 0:2]
    z_hat = torch.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
    # Construct measurement Jacobian
    jacobian_h = torch.zeros((len(z_hat), 1, 3), dtype=torch.float64, device=landmarks.device)
    # pdb.set_trace()
    jacobian_h[:, :, 0] = (-q[:, 0] / z_hat).unsqueeze(1)
    jacobian_h[:, : , 1] = (-q[:, 1] / z_hat).unsqueeze(1)
    
    posterior_mean, posterior_cov = ekf_filter.update(range_beacon.squeeze(1), measurements, z_hat.unsqueeze(-1) , jacobian_h, measurement_cov_)
    nll_posterior = ekf_filter.neg_log_likelihood(inputs, posterior_mean, posterior_cov)
    return posterior_mean, posterior_cov, nll_posterior


def train_range_ekf_step(model, inputs, measurements, control, ekf_filter, MOTION_NOISE):

    with torch.no_grad():
        motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
        motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
        # measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
        state_pred, cov_pred = ekf_filter.prediction(control, motion_model_cov_)


    z_hat, z_cov, predicted_jacobian = model(state_pred.to(torch.float32).to(device))

    z_hat_ = z_hat.to(torch.float64) #[batch_size, 1]
    predicted_jacobian_ = predicted_jacobian.to(torch.float64) #[batch_size, 1, 3]
    z_cov_ = torch.diag_embed(z_cov.to(torch.float64)) #[1, 1]
    posterior_mean, posterior_cov = ekf_filter.update(measurements, z_hat_ , predicted_jacobian_,z_cov_)
    nll_posterior = ekf_filter.neg_log_likelihood(inputs, posterior_mean, posterior_cov)

    with torch.no_grad():
        nll_measurement_likelihood = ekf_filter.neg_log_likelihood(measurements, z_hat_, z_cov_)
    
    return posterior_mean, posterior_cov, nll_posterior, nll_measurement_likelihood, z_hat_


# class ObservationModelNN(nn.Module):
#     def __init__(self, state_dim, measurement_dim):
#         """
#         Neural Network to predict the observation mean, covariance, and Jacobian,
#         conditioned on both the state and the measurement.

#         Args:
#             state_dim (int): The dimension of the state space (e.g., 3 for [x, y, theta]).
#             measurement_dim (int): The dimension of the measurement space (e.g., 1 for distance).
#         """
#         super(ObservationModelNN, self).__init__()
        
#         # Now input dimension = state_dim + measurement_dim
#         input_dim = state_dim + measurement_dim

#         self.hidden = nn.Sequential(
#             nn.Linear(input_dim, 128),  # First hidden layer
#             nn.ReLU(),
#             nn.Linear(128, 64),  # Second hidden layer
#             nn.ReLU()
#         )
        
#         # Output layers
#         self.obs_output = nn.Linear(64, measurement_dim)      # Predicted observation mean
#         self.cov_output = nn.Linear(64, measurement_dim)      # Predicted log-variance
#         self.jacobian_output = nn.Linear(64, 3)               # Predicted Jacobian (1x3)

#     def forward(self, state, measurement):
#         """
#         Forward pass of the neural network.

#         Args:
#             state (torch.Tensor): State tensor of shape (batch_size, state_dim).
#             measurement (torch.Tensor): Measurement tensor of shape (batch_size, measurement_dim).

#         Returns:
#             torch.Tensor: Predicted observation (batch_size, measurement_dim).
#             torch.Tensor: Predicted covariance (batch_size, measurement_dim).
#             torch.Tensor: Predicted Jacobian (batch_size, 1, 3).
#         """
#         # Concatenate state and measurement along the feature dimension
#         x = torch.cat([state, measurement], dim=-1)  # Shape (batch_size, state_dim + measurement_dim)
        
#         hidden_out = self.hidden(x)

#         obs_pred = self.obs_output(hidden_out)
#         log_cov_pred = self.cov_output(hidden_out)
#         log_cov_pred = torch.clamp(log_cov_pred, min=-10.0, max=10.0)
#         cov_pred = torch.exp(log_cov_pred)  # Ensure positivity
        
#         jacobian_pred = self.jacobian_output(hidden_out).view(-1, 1, 3)
        
#         return obs_pred, cov_pred, jacobian_pred


class ObservationModelNN(nn.Module):
    def __init__(self, state_dim, measurement_dim):
        """
        Neural Network to predict the observation, observation covariance, and Jacobian (1x3).
        
        Args:
            state_dim (int): The dimension of the state space (e.g., 3 for [x, y, theta]).
            measurement_dim (int): The dimension of the measurement space (e.g., 1 for distance).
        """
        super(ObservationModelNN, self).__init__()
        
        # Define the hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, 128),  # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU()
        )
        
        # Output layer for predicted observation (mean)
        self.obs_output = nn.Linear(64, measurement_dim)
        
        # Output layer for predicted covariance (log of diagonal elements)
        self.cov_output = nn.Linear(64, measurement_dim)  # Predict log of variance
        
        # Output layer for Jacobian (1x3 for each measurement)
        self.jacobian_output = nn.Linear(64, 3)  # Predict Jacobian with 3 components (1x3 for each measurement)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input state vector (batch_size, state_dim).
        
        Returns:
            torch.Tensor: Predicted observation (batch_size, measurement_dim).
            torch.Tensor: Log of predicted covariance diagonal (batch_size, measurement_dim).
            torch.Tensor: Predicted Jacobian (batch_size, 1, 3).
        """
        # Pass through hidden layers
        hidden_out = self.hidden(x)
        
        # Predicted observation (mean)
        obs_pred = self.obs_output(hidden_out)
        
        # Predicted covariance (log of variances to ensure positivity)
        log_cov_pred = self.cov_output(hidden_out)
        log_cov_pred = torch.clamp(log_cov_pred, min=-10.0, max=10.0)
        cov_pred = torch.exp(log_cov_pred)  # Convert log to actual covariance (variance)
        
        # Predicted Jacobian (1x3 for each measurement)
        jacobian_pred = self.jacobian_output(hidden_out).view(-1, 1, 3)  # Reshape to (batch_size, 1, 3)
        
        return obs_pred, cov_pred, jacobian_pred


def training_ekf(loggin_path, args, model_path):
    poses, X, Y, T = se2_grid_samples_torch(args.batch_size, args.grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    decay_rate = args.decay_rate 
    STEP_MOTION = SE2Group(args.step_motion[0], args.step_motion[1], args.step_motion[2])
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    MEASUREMENT_NOISE = np.sqrt(args.measurement_cov)
    # Generate dataset
    train_loader, val_loader, _ = generate_bounded_se2_dataset(
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        step_motion=STEP_MOTION,
        motion_noise=MOTION_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
        samples=poses.cpu().numpy(),
        batch_size=args.batch_size, 
        validation_split=args.validation_split,
        test_split=args.test_split
    )
    cov_prior = torch.diag(torch.tensor(args.cov_prior, dtype=torch.float64)).to(device)
    cov_prior_batch = torch.tile(cov_prior.unsqueeze(0), [args.batch_size, 1, 1])
    
    # Example usage
    state_dim = 3  # Example: 3D state (x, y, theta)
    measurement_dim = 1  # Example: 1D measurement (e.g., distance)

    # Create the model
    model = ObservationModelNN(state_dim, measurement_dim).to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        epoch_start = int(model_path.split("_")[-1].split(".")[0])
    else:
        initialize_weights(model)
        epoch_start = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    torch.autograd.set_detect_anomaly(True)
    lr_decay = lambda epoch : (args.learning_rate_end / args.learning_rate_start) ** ((epoch / args.num_epochs) * args.slope_weight)
    for epoch in range(epoch_start, args.num_epochs):
        model.train()
        total_loss = 0
        total_rmse = 0
        total_nll_posterior = 0
        total_nll_likelihood = 0
        validate_ekf(model, epoch, val_loader, poses, X, Y, T, args)
        new_lr = args.learning_rate_start * lr_decay(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # if epoch < args.threshold_warmup:
        regularizer_weight_warmup = math.exp(-args.decay_rate * ((epoch) / (args.threshold_warmup)))
        # else: 
        #     regularizer_weight = math.exp(-args.decay_rate * ((epoch - args.threshold_warmup) / (args.num_epochs - args.threshold_warmup)))
        
        for data in train_loader:
            inputs, measurements, control, range_beacon = data
            inputs, measurements, control, range_beacon = inputs.to(device), measurements.to(device), control.to(device), range_beacon.to(device)
            ekf_filter = RangeEKF(inputs[:, 0], cov_prior_batch)
            trajectory_list = []

            for i in range(args.trajectory_length - 1):
                traj_idx = i + 1
                posterior_mean, posterior_cov, nll_posterior, nll_measurement_likelihood, predicted_measurement = train_range_ekf_step(model, 
                    inputs[:, traj_idx], measurements[:, i], control[:, i],
                    ekf_filter, MOTION_NOISE)
                
                # if epoch < args.threshold_warmup:
                # loss = (regularizer_weight_warmup * torch.mean(torch.abs(predicted_measurement - measurements[:,i])) + (1-regularizer_weight_warmup) * nll_measurement_likelihood).to(torch.float32)
                # loss = 0.5*(nll_posterior.to(torch.float32) + mse(posterior_mean, inputs[:, traj_idx]).to(torch.float32))
                loss = nll_posterior.to(torch.float32)

                #     loss = (regularizer_weight * nll_measurement_likelihood + (1 - regularizer_weight) * F.mse_loss(posterior_mean ,inputs[:, traj_idx])).to(torch.float32)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                total_nll_likelihood += float(nll_measurement_likelihood.detach().item())
                total_nll_posterior += float(nll_posterior.detach().item())
                trajectory_list.append(posterior_mean)

            predicted_trajectory = torch.stack(trajectory_list, dim=1)
            total_rmse += rmse_se2(inputs, predicted_trajectory)

        avg_loss = total_loss / (len(train_loader) * (args.trajectory_length - 1))
        avg_nll_posterior = total_nll_posterior / (len(train_loader) * (args.trajectory_length - 1))
        avg_nll_likelihood = total_nll_likelihood / (len(train_loader) * (args.trajectory_length - 1))
        avg_rmse = total_rmse / len(train_loader)

        wandb.log({
            "Training Loss": avg_loss,
            "Training RMSE": avg_rmse,
            "Training NLL Posterior": avg_nll_posterior,
            "Training NLL Likelihood": avg_nll_likelihood,
            "Epoch": epoch,
        })

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, NLL Measurement: {avg_nll_likelihood:.4f}, NLL Posterior: {avg_nll_posterior:.4f}")

        if (epoch+1) % 20 == 0:
            # Save the model every 20 epochs
            model_save_path = os.path.join(logging_path, f"observation_ekf_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path, base_path=logging_path)

def validate_ekf(model, epoch, val_loader, poses, X, Y, T, args):
    model.eval()
    total_rmse = 0
    total_nll_posterior = 0
    l1_measurement = 0
    total_nll_likelihood = 0
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    MEASUREMENT_NOISE = np.sqrt(args.measurement_cov)
    cov_prior = torch.diag(torch.tensor(args.cov_prior, dtype=torch.float64)).to(device)
    cov_prior_batch = torch.tile(cov_prior.unsqueeze(0), [args.batch_size, 1, 1])

    with torch.no_grad():
        for data in val_loader:
            inputs, measurements, control, range_beacon = data
            inputs, measurements, control, range_beacon = inputs.to(device), measurements.to(device), control.to(device), range_beacon.to(device)

            ekf_filter = RangeEKF(inputs[:, 0], cov_prior_batch)
            trajectory_list = []

            for i in range(args.trajectory_length - 1):
                traj_idx = i + 1
                posterior_mean, posterior_cov, nll_posterior, nll_measurement_likelihood, predicted_measurement = train_range_ekf_step(model, 
                    inputs[:, traj_idx], measurements[:, i], control[:, i],
                    ekf_filter, MOTION_NOISE)
                
                l1_measurement += torch.mean(torch.abs(predicted_measurement - measurements[:, i]))
                total_nll_posterior += float(nll_posterior.item())
                total_nll_likelihood += float(nll_measurement_likelihood.item())
                trajectory_list.append(posterior_mean)

            predicted_trajectory = torch.stack(trajectory_list, dim=1)
            total_rmse += rmse_se2(inputs, predicted_trajectory)

    avg_rmse = total_rmse / len(val_loader)
    avg_nll_posterior = total_nll_posterior / (len(val_loader) * (args.trajectory_length - 1))
    avg_nll_likelihood = total_nll_likelihood / (len(val_loader) * (args.trajectory_length - 1))
    avg_l1_measurement = l1_measurement / (len(val_loader) * (args.trajectory_length - 1))

    wandb.log({
        "Validation RMSE": avg_rmse,
        "Validation NLL Posterior": avg_nll_posterior,
        "Validation NLL Likelihood": avg_nll_likelihood,
        "Validation L1 Measurement": avg_l1_measurement,
        "Epoch": epoch,
    })

    print(f"Validation RMSE: {avg_rmse:.4f}, NLL Posterior: {avg_nll_posterior:.4f}, NLL Likelihood: {avg_nll_likelihood:.4f}, L1 Measurement: {avg_l1_measurement:.4f}")

import ast

def parse_list(value):
    """Helper function to parse a list from string input."""
    try:
        return ast.literal_eval(value)  # Safely convert the string into a Python literal (e.g., list)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
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
    parser.add_argument('--seed', type=int, default=4589, help='Random seed')
    parser.add_argument('--decay_rate', type=float, default=5, help='Decay rate for regularization')
    parser.add_argument('--threshold_warmup', type=int, default=200, help='Threshold for warmup')
    parser.add_argument('--learning_rate_start', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Final learning rate')
    parser.add_argument('--slope_weight', type=float, default=1, help='Slope weight for learning rate decay')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # run = wandb.init(project="Diff-HEF",group="SE2",entity="korra141",
    #           tags=["SE2","DenistyEstimation","UnimodalNoise","Training"],
    #           name="SE2-DiffHEF-RangeSimulator-1",
    #           notes="Diff-HEF on SE2 Range Simulator",
    #           config=args)
    run = wandb.init(project="Diff-EKF",group="SE2",entity="korra141",
              tags=["SE2", "NLLPosterior","ModelwithMeanBelief"],
              name="SE2-DiffEKF-RangeSimulator-1",
              notes="Diff-EKF on SE2 Range Simulator",
              config=args)
    run_name = "SE2_Range_EKF"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
    model_path = None
    training_ekf(logging_path, args, model_path)