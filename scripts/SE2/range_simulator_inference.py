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
from torch.distributions import Normal
import ast
import json

base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)
pid = os.getpid()

torch.cuda.empty_cache()
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.R1.HarmonicExponentialDistribution import HarmonicExponentialDistribution as R1_HED
from src.distributions.SE2.SE2_torch import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from src.filter.HEF_SE2 import HEFilter
from src.distributions.SE2.se2_distribution import SE2, SE2Gaussian
from src.filter.bayes_filter import BayesFilter
from src.filter.EKF import RangeEKF
from src.filter.HistF import BatchedRangeHF
from src.filter.PF import RangePF
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean, mse
from src.utils.visualisation import plot_se2_mean_filters,plot_se2_filters
import argparse
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMFilterSE2(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, state_dim=3):
        super(LSTMFilterSE2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # Fully connected layers to map hidden state to state and covariance
        self.state_fc = nn.Linear(self.hidden_dim, self.state_dim)
        self.log_var_fc = nn.Linear(self.hidden_dim, self.state_dim)

    def forward(self, control_inputs, measurements, initial_state, cov_prior):
        inputs = torch.cat([control_inputs, measurements], dim=-1)  # Combine control and measurements
        lstm_out, _ = self.lstm(inputs)  # Pass through LSTM
        estimated_states = self.state_fc(lstm_out)  # Predict states
        log_var = self.log_var_fc(lstm_out)  # Predict log-variance
        state_result = torch.cat((initial_state.unsqueeze(1), estimated_states), dim=1)
        state_result[..., 2] = (state_result[..., 2] + math.pi) % (2 * math.pi) - math.pi  # Wrap angle
        cov = torch.exp(log_var)
        cov_result = torch.cat((cov_prior.unsqueeze(1), cov), dim=1)
        return state_result, cov_result

# class DensityEstimatorHistF(nn.Module):
#     def __init__(self, input_size):
#         super(DensityEstimatorHistF, self).__init__()
#         self.fc1 = nn.Linear(input_size, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, input_size)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
#         self.softplus = nn.Softplus()

#     def forward(self, x):
#         x = self.leaky_relu(self.fc1(x))
#         x = self.leaky_relu(self.fc2(x))
#         x = self.softplus(self.fc3(x))
#         return x


class DensityEstimatorHistF(nn.Module):
    def __init__(self, input_size, measurement_size=1):
        super(DensityEstimatorHistF, self).__init__()
        
        # Now input includes both state and measurement
        total_input_size = input_size + measurement_size

        self.fc1 = nn.Linear(total_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, input_size)  # Output stays same size as input (e.g., grid density)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softplus = nn.Softplus()

    def forward(self, x, measurement):
        """
        Args:
            x: Tensor of shape [batch_size, input_size]
            measurement: Tensor of shape [batch_size, 1]
        """
        # Concatenate along last dimension
        combined_input = torch.cat([x, measurement], dim=-1)  # shape: [batch, input_size + 1]

        x = self.leaky_relu(self.fc1(combined_input))
        x = self.leaky_relu(self.fc2(x))
        x = self.softplus(self.fc3(x))

        return x

# class ObservationModelEKF(nn.Module):
#     def __init__(self, state_dim, measurement_dim):
#         """
#         Neural Network to predict the observation mean, covariance, and Jacobian,
#         conditioned on both the state and the measurement.

#         Args:
#             state_dim (int): The dimension of the state space (e.g., 3 for [x, y, theta]).
#             measurement_dim (int): The dimension of the measurement space (e.g., 1 for distance).
#         """
#         super(ObservationModelEKF, self).__init__()
        
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

class ObservationModelEKF(nn.Module):

    def __init__(self, state_dim, measurement_dim):
        """
        Neural Network to predict the observation, observation covariance, and Jacobian (1x3).
        
        Args:
            state_dim (int): The dimension of the state space (e.g., 3 for [x, y, theta]).
            measurement_dim (int): The dimension of the measurement space (e.g., 1 for distance).
        """
        super(ObservationModelEKF, self).__init__()
        
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


# class ObservationModelEKF(nn.Module):
#     def __init__(self, state_dim, measurement_dim):
#         """
#         Neural Network to predict the observation, observation covariance, and Jacobian (1x3).
        
#         Args:
#             state_dim (int): The dimension of the state space (e.g., 3 for [x, y, theta]).
#             measurement_dim (int): The dimension of the measurement space (e.g., 1 for distance).
#         """
#         super(ObservationModelEKF, self).__init__()
        
#         # Define the hidden layers
#         self.hidden = nn.Sequential(
#             nn.Linear(state_dim, 128),  # First hidden layer
#             nn.ReLU(),
#             nn.Linear(128, 64),  # Second hidden layer
#             nn.ReLU()
#         )
        
#         # Output layer for predicted observation (mean)
#         self.obs_output = nn.Linear(64, measurement_dim)
        
#         # Output layer for predicted covariance (log of diagonal elements)
#         self.cov_output = nn.Linear(64, measurement_dim)  # Predict log of variance
        
#         # Output layer for Jacobian (1x3 for each measurement)
#         self.jacobian_output = nn.Linear(64, 3)  # Predict Jacobian with 3 components (1x3 for each measurement)

#     def forward(self, x):
#         """
#         Forward pass of the neural network.

#         Args:
#             x (torch.Tensor): The input state vector (batch_size, state_dim).
        
#         Returns:
#             torch.Tensor: Predicted observation (batch_size, measurement_dim).
#             torch.Tensor: Log of predicted covariance diagonal (batch_size, measurement_dim).
#             torch.Tensor: Predicted Jacobian (batch_size, 1, 3).
#         """
#         # Pass through hidden layers
#         hidden_out = self.hidden(x)
        
#         # Predicted observation (mean)
#         obs_pred = self.obs_output(hidden_out)
        
#         # Predicted covariance (log of variances to ensure positivity)
#         log_cov_pred = self.cov_output(hidden_out)

#         cov_pred = torch.exp(log_cov_pred)  # Convert log to actual covariance (variance)
        
#         # Predicted Jacobian (1x3 for each measurement)
#         jacobian_pred = self.jacobian_output(hidden_out).view(-1, 1, 3)  # Reshape to (batch_size, 1, 3)
        
#         return obs_pred, cov_pred, jacobian_pred

def lstm_nll(predicted_state, cov,  true_state):
    diff = predicted_state - true_state #[batch_size, trajectory_length, state_dim]
    sigma = torch.sqrt(cov) #[batch_size, trajectory_length, state_dim]
    nll = 0.5 * torch.sum((diff ** 2) / (sigma ** 2) + torch.log(sigma ** 2) + torch.log(torch.tensor(2) * torch.pi), dim=2)
    
    return torch.mean(nll)

def range_hist_step(inputs, measurements, control, range_beacon, hist_filter, MOTION_NOISE, MEASUREMENT_NOISE):

    batch_size = inputs.shape[0]
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    # measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
    # measurement_cov_ = torch.tile(measurement_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    hist_filter.prediction(control, motion_model_cov_)
    dist = torch.linalg.norm(range_beacon - hist_filter.grid_samples[:, :, 0:2], dim=-1)
    energy = torch.distributions.Normal(measurements, MEASUREMENT_NOISE).log_prob(dist)
    # Normalize log likelihoods
    energy -= torch.logsumexp(energy, dim=-1, keepdim=True)
    posterior_mean, posterior_pdf = hist_filter.update(energy)
    nll_posterior = hist_filter.neg_log_likelihood(inputs, posterior_pdf.view(batch_size, -1))
    return posterior_mean, posterior_pdf , nll_posterior

def train_hist_step(range_beacon, model, hist_filter, inputs, measurements, control, args):
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    with torch.no_grad():
        belief_hat = hist_filter.prediction(control,  motion_model_cov_)
    # Measurement model using the provided model
    measurement_pdf = model(belief_hat.to(torch.float32),measurements.to(torch.float32))
    energy = torch.log(measurement_pdf.to(torch.float64) + 1e-8)
    energy_ = energy - torch.logsumexp(energy, dim=1, keepdim=True)  # Normalize log likelihoods

    # with torch.no_grad():
    posterior_mean_hist, posterior_pdf_hist = hist_filter.update(energy_)

    nll_posterior_hist = hist_filter.neg_log_likelihood(inputs,posterior_pdf_hist.view(args.batch_size, -1) )
    nll_measurement_hist = hist_filter.neg_log_likelihood_measurement(range_beacon, measurements, energy_)

    return posterior_mean_hist, posterior_pdf_hist, nll_posterior_hist, nll_measurement_hist

def range_ekf_step(inputs, measurements, control, range_beacon, ekf_filter, pose, MOTION_NOISE, MEASUREMENT_NOISE):
    start_time = datetime.datetime.now()
    # Prediction step
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
    batch_size = inputs.shape[0]
    measurement_cov_ = torch.tile(measurement_cov.unsqueeze(0), [batch_size , 1,  1])
    predicted_pose , predicted_cov = ekf_filter.prediction(control, motion_model_cov_)
    q = range_beacon.squeeze(1) - predicted_pose[:,0:2]
    z_hat = torch.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
    # Construct measurement Jacobian
    jacobian_h = torch.zeros((len(z_hat), 1, 3), dtype=torch.float64, device=range_beacon.device)
    # pdb.set_trace()
    jacobian_h[:, :, 0] = (-q[:, 0] / z_hat).unsqueeze(1)
    jacobian_h[:, : , 1] = (-q[:, 1] / z_hat).unsqueeze(1)
    
    posterior_mean, posterior_cov = ekf_filter.update(measurements, z_hat.unsqueeze(-1) , jacobian_h, measurement_cov_)
    nll_posterior = ekf_filter.neg_log_likelihood(inputs, posterior_mean, posterior_cov)
    return posterior_mean, posterior_cov, nll_posterior


def train_range_ekf_step(model, inputs, measurements, control, ekf_filter, MOTION_NOISE):

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

    nll_measurement_likelihood = ekf_filter.neg_log_likelihood(measurements, z_hat_, z_cov_)
    
    return posterior_mean, posterior_cov, nll_posterior, nll_measurement_likelihood, z_hat_

def analytic_hef(inputs, measurements, range_beacon, control, poses, X, Y, T,  grid_size, batch_size, hef_filter, hed_r1, fft, motion_cov ,motion_inv_cov, MEASUREMENT_NOISE):

    motion_model = SE2Gaussian(control, motion_cov, motion_inv_cov, samples=poses, fft=fft)
    motion_model.normalize()

    belief_hat = hef_filter.prediction(motion_model)

    dist = torch.linalg.norm(poses[:, :, 0:2] - range_beacon, dim=-1)
    energy = torch.distributions.Normal(measurements, MEASUREMENT_NOISE).log_prob(dist)
    energy = energy.reshape(-1, *args.grid_size)
    range_prob = torch.exp(energy)
    # Calculate log-likelihood with numerical stability
    range_ll = torch.log(range_prob + 1e-8)
    measurement_model = SE2(samples=poses,fft=fft)

    _, _, _, _, _, measurement_model.eta = fft.analyze(range_ll)
    measurement_model.energy, _, _, _, _, _ = fft.synthesize(measurement_model.eta)
    measurement_model.normalize()

    # measurement_density_flat = measurement_model.prob.reshape(batch_size, -1)
    
    posterior = hef_filter.update(measurement_model)

    nll_posterior = torch.mean(fft.neg_log_likelihood(posterior.energy, inputs))

    predicted_pose = compute_weighted_mean(posterior.prob, poses, X, Y, T)

    dist = torch.linalg.norm(range_beacon  - poses[:, :, 0:2], dim=-1)
    range_L = dist.max(dim=-1).values - dist.min(dim=-1).values
    range_L = range_L.view(-1, 1, 1)
    measurement_energy = measurement_model.energy.reshape(batch_size, -1)
    nll_measurement_likelihood = hed_r1.negative_log_likelihood(measurement_energy, measurements, range_L)

    return posterior, measurement_model , belief_hat,  predicted_pose, nll_measurement_likelihood, nll_posterior

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

def normalize_angle(input_):
    output = input_.clone()
    output[:, 2] = (output[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return output 

def inference(args, logging_path,  diff_hef_model_path=None,  diff_ekf_model_path=None,  diff_hist_model_path=None,  diff_lstm_model_path=None):
    
    flag_visual_logging = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    if diff_hef_model_path and os.path.exists(diff_hef_model_path):
        model_hef = DensityEstimator(args.grid_size).to(device)
        model_hef.load_state_dict(torch.load(diff_hef_model_path))
        model_hef.eval()
    else:
        print("Diff_HEF Model path is not provided or does not exist.")

    if diff_ekf_model_path and os.path.exists(diff_ekf_model_path):
        state_dim = 3  # Example: 3D state (x, y, theta)
        measurement_dim = 1  # Example: 1D measurement (e.g., distance)
        model_ekf = ObservationModelEKF(state_dim, measurement_dim).to(device)
        model_ekf.load_state_dict(torch.load(diff_ekf_model_path))
        model_ekf.eval()
    else:
        print("Diff_EKF Model path is not provided or does not exist.")

    if diff_hist_model_path and os.path.exists(diff_hist_model_path):
        model_hist = DensityEstimatorHistF(math.prod(args.grid_size)).to(device)
        model_hist.load_state_dict(torch.load(diff_hist_model_path))
        model_hist.eval()
    else:
        print("Diff_HistF Model path is not provided or does not exist.")

    if diff_lstm_model_path and os.path.exists(diff_lstm_model_path):
        lstm_filter =  LSTMFilterSE2().to(device)
        lstm_filter.load_state_dict(torch.load(diff_lstm_model_path))
        lstm_filter.eval()
    else:
        print("LSTM Model path is not provided or does not exist.")

    range_x = (-0.5, 0.5)
    range_y = (-0.5, 0.5)
    fft = SE2_FFT(spatial_grid_size=args.grid_size,
                       interpolation_method='spline',
                       spline_order=args.spline_order,
                       oversampling_factor=3, 
                       device=device)
    TRUE_MEASUREMENT_NOISE = np.sqrt(args.measurement_cov).item()
    # ESTIMATED_MEASUREMENT_NOISE = np.sqrt(args.estimated_measurement_cov).item()
    ESTIMATED_MEASUREMENT_NOISE = TRUE_MEASUREMENT_NOISE
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))

    hed_r1 = R1_HED(math.prod(args.grid_size), torch.sqrt(torch.tensor(2)))

    poses, X, Y, T = se2_grid_samples_torch(args.batch_size, args.grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    
    legend = [ rf"Predicted belief", rf"Measurement", rf"Posterior"]
    CONFIG_MEAN_SE2_LF = [
    # {'label': 'Diff-HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
    #  'alpha': 0.8},
    {'label': 'HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, 'cmap': plt.cm.Greens,
     'alpha': 0.8},
    # {'label': 'Estimated-HEF', 'c': '#9467bd', 'marker': '<', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
    #  'alpha': 0.8},
    {'label': 'EKF', 'c': '#d62728', 'marker': 'D', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, 'cmap': plt.cm.Reds,
     'alpha': 0.8},
    {'label': 'PF', 'c': '#9467bd', 'marker': '<', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, 'cmap': plt.cm.Purples,
     'alpha': 0.8},
    {'label': 'HistF', 'c': '#8c564b', 'marker': 'p', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, 'cmap': plt.cm.pink_r,
     'alpha': 0.8},
    {'label': 'GT', 'c': '#e377c2', 'marker': '*', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
     'zorder': 4, 'alpha': 0.8},
    {'label': 'Beacons', 'c': 'dimgrey', 'marker': 'o', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
     'zorder': 2, 'alpha': 0.8}
    #  {'label': 'Diff-HistF', 'c': '#2ca02c', 'marker': 's', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, 'cmap': plt.cm.Oranges,
    #  'alpha': 0.8},
    #  {'label': 'Diff-EKF', 'c': '#9467bd', 'marker': 's', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, "cmap": plt.cm.Blues, "alpha": 0.8},
    #  {'label': 'LSTM', 'c': '#8c564b', 'marker': 's', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3, "cmap": plt.cm.Greys, "alpha": 0.8}
    ]

    train_loader , val_loader , test_loader = generate_bounded_se2_dataset(
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        step_motion=SE2Group(args.step_motion[0], args.step_motion[1], args.step_motion[2]),
        motion_noise=MOTION_NOISE,
        measurement_noise=TRUE_MEASUREMENT_NOISE,
        samples=poses.cpu().numpy(),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split = args.test_split
    )

    # data = next(iter(test_loader))
    # inputs, measurements,  control, beacon_idx  = data
    # # Save the data to a .npz file
    # save_path = os.path.join(logging_path, "dataset.npz")
    # np.savez(save_path, 
    #          true_trajectories=inputs.cpu().numpy(),
    #          measurements=measurements.cpu().numpy(),
    #          control=control.cpu().numpy(),
    #          beacond_idx=beacon_idx.cpu().numpy())
    # print(f"Data saved successfully to: {save_path}")
    # return
    
    # Load the .npz file containing the dictionary
    # data_path = "/home/mila/r/ria.arora/scratch/local/HarmonicExponentialBayesFitler/dataset.npz"
    # if os.path.exists(data_path):
    #     data = np.load(data_path, allow_pickle=True)
    #     data_dict = {key: data[key] for key in data.files}
    #     print("Data loaded successfully from:", data_path)
    # else:
    #     raise FileNotFoundError(f"The specified data file does not exist: {data_path}")

    # inputs = torch.tensor(data_dict["true_trajectories"]).unsqueeze(0).to(device).to(torch.float64)
    # measurements = torch.tensor(data_dict["measurements"]).unsqueeze(0).to(device).to(torch.float64)
    # measurements_energy = torch.tensor(data_dict["measurements_energy"]).unsqueeze(0).to(device).to(torch.float64)
    # control = torch.tensor(data_dict["control"]).unsqueeze(0).to(device).to(torch.float64)
    # beacon_idx = torch.tensor(data_dict["beacond_idx"]).unsqueeze(0).to(torch.long).to(device)
    # print("NPZ loaded into tensors.")

    beacons = torch.tensor(
            [[0, 0.1],
             [0, 0.05],
             [0, 0.0],
             [0, -0.05],
             [0, -0.1]]).to(device)
    
    cov_prior = torch.diag(torch.tensor(args.cov_prior)).to(torch.float64).to(device)
    cov_prior_batch = torch.tile(cov_prior.unsqueeze(0), [args.batch_size, 1, 1])
    inv_cov_prior = torch.inverse(cov_prior).to(device)

    diag_cov_prior_batch = torch.tile(torch.tensor(args.cov_prior).unsqueeze(0),[args.batch_size, 1]).to(device)

    motion_cov = torch.diag(torch.tensor(args.motion_cov)).to(torch.float64).to(device)
    motion_inv_cov = torch.inverse(motion_cov).to(device)

    with torch.no_grad():
        total_rmse_true = 0
        total_rmse_ekf = 0
        total_rmse_hist = 0
        total_rmse_diff_ekf = 0
        total_rmse_diff_hist = 0
        total_rmse_pf = 0
        total_rmse_lstm = 0


        total_nll_likelihood_true = 0
        total_nll_posterior_true = 0
        total_nll_posterior_ekf = 0
        total_nll_posterior_diff_ekf = 0
        total_nll_posterior_hist = 0
        total_nll_posterior_diff_hist = 0
        total_nll_posterior_pf = 0
        total_nll_posterior_lstm = 0

        num_samples = 0
        sample_batch = 0
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
        # for i in range(1):
            # if batch_idx > 0:
            #     break
            # print(f"Batch {batch_idx}/{len(train_loader)}")
            num_samples += 1
            inputs, measurements,  control, beacon_idx  = data
            inputs, measurements,  control, beacon_idx = inputs.to(device), measurements.to(device), control.to(device), beacon_idx.to(device)
            prior = SE2Gaussian(inputs[:, 0], cov_prior, inv_cov_prior, samples=poses, fft=fft)
            prior.normalize()
            
            true_hef_filter = BayesFilter(distribution=SE2, prior=prior, device=device)
            
            ekf_filter = RangeEKF(inputs[:, 0], cov_prior_batch)

            diff_ekf_filter = RangeEKF(inputs[:, 0], cov_prior_batch)

            hist_filter = BatchedRangeHF(args.batch_size, inputs[:,0],cov_prior_batch, poses, X, Y, T,grid_size=args.grid_size, device=device)

            diff_hist_filter = BatchedRangeHF(args.batch_size, inputs[:, 0], cov_prior_batch, poses, X, Y, T, grid_size=args.grid_size, device=device)
        
            pf_filter = RangePF(inputs[:, 0],cov_prior_batch, poses.shape[1], args.batch_size, args.grid_size, device=device)

            trajectory_list_true = []
            trajectory_list_ekf = []
            trajectory_list_diff_ekf = []
            trajectory_list_hist = []
            trajectory_list_diff_hist = []
            trajectory_list_pf =[]

            trajectory_list_ekf.append(inputs[:, 0])
            trajectory_list_true.append(inputs[:, 0])
            trajectory_list_hist.append(inputs[:, 0])
            trajectory_list_diff_hist.append(inputs[:, 0])
            trajectory_list_diff_ekf.append(inputs[:, 0])
            trajectory_list_pf.append(inputs[:, 0])
           
            sample_index = random.randint(0, args.batch_size-1)
            for i in range(args.trajectory_length-1):  # Iterate over trajectory length
                traj_idx = i + 1
                # posterior, measurement_model, belief_hat, predicted_pose, nll_measurement_likelihood, nll_posterior = diff_hef_step(inputs[:, traj_idx], measurements[:, traj_idx], control[:, traj_idx], poses, X, Y, T, grid_size, diff_hef_filter, model, fft, hed_r1, MOTION_NOISE, args.batch_size)
                
                # a filter here for analytic with true noise 
                range_beacon = beacons[beacon_idx[:, i ,0:1], :]
                # beacons[beacon_idx[:, i], :]
                
                # HARMONIC EXPONENTIAL FILTER
                posterior_true, measurement_model_true, belief_hat_true, predicted_pose_true, nll_measurement_likelihood_true, nll_posterior_true = analytic_hef(
                    inputs[:, traj_idx], measurements[:, i], range_beacon, control[:, i], poses, X, Y, T, args.grid_size, args.batch_size, true_hef_filter, hed_r1, fft, motion_cov ,motion_inv_cov, ESTIMATED_MEASUREMENT_NOISE)


                trajectory_list_true.append(predicted_pose_true)
                total_nll_likelihood_true += float(nll_measurement_likelihood_true.item())
                total_nll_posterior_true += float(nll_posterior_true.item())

                # # a filter here for analytic with estimated noise 
                # posterior_estimated, measurement_model_estimated, belief_hat_estimated, motion_model_estimated,  predicted_pose_estimated, nll_measurement_likelihood_estimated, nll_posterior_estimated = analytic_hef(
                #     inputs[:, traj_idx], measurements[:, i], range_beacon, control[:, i], poses, X, Y, T, args.grid_size, args.batch_size, analytic_hef_filter, hed_r1, fft, MOTION_NOISE, ESTIMATED_MEASUREMENT_NOISE)
                
                # HISTOGRAM FILTER
                posterior_mean_hist, posterior_pdf_hist, nll_posterior_hist = range_hist_step(
                    inputs[:, traj_idx], measurements[:, i], control[:, i], range_beacon, 
                    hist_filter, MOTION_NOISE=MOTION_NOISE, MEASUREMENT_NOISE=ESTIMATED_MEASUREMENT_NOISE)

                trajectory_list_hist.append(posterior_mean_hist)
                total_nll_posterior_hist += float(nll_posterior_hist.item())

                if diff_hist_model_path is not None:
                    posterior_mean_diff_hist, posterior_pdf_diff_hist, nll_posterior_diff_hist, nll_measurement_likelihood_diff_hist = train_hist_step(
                        range_beacon, model_hist, diff_hist_filter, inputs[:, traj_idx], measurements[:, i], control[:, i], args
                    )
                    trajectory_list_diff_hist.append(posterior_mean_diff_hist)
                    total_nll_posterior_diff_hist += float(nll_posterior_diff_hist.item())

                # EXTENDED KALMAN FILTER
                posterior_mean_ekf, posterior_cov_ekf, nll_posterior_ekf = range_ekf_step(
                    inputs[:, traj_idx], measurements[:, i], control[:, i], range_beacon, 
                    ekf_filter, poses , MOTION_NOISE=MOTION_NOISE, MEASUREMENT_NOISE=ESTIMATED_MEASUREMENT_NOISE)

                trajectory_list_ekf.append(posterior_mean_ekf)
                total_nll_posterior_ekf += float(nll_posterior_ekf.item())

                if diff_ekf_model_path is not None:

                    posterior_mean_diff_ekf, posterior_cov_diff_ekf, nll_posterior_diff_ekf, nll_measurement_likelihood_diff_ekf, predicted_measurement_diff_ekf = train_range_ekf_step(model_ekf, 
                        inputs[:, traj_idx], measurements[:, i], control[:, i],diff_ekf_filter, MOTION_NOISE)

                    trajectory_list_diff_ekf.append(posterior_mean_diff_ekf)
                    total_nll_posterior_diff_ekf += float(nll_posterior_diff_ekf.item())

                # PARTICLE FILTER
                posterior_mean_pf, posterior_pf, nll_posterior_pf = range_pf_step(inputs[:, traj_idx], measurements[:, i], control[:, i], range_beacon,
                    pf_filter, MOTION_NOISE=MOTION_NOISE, MEASUREMENT_NOISE=ESTIMATED_MEASUREMENT_NOISE)

                trajectory_list_pf.append(posterior_mean_pf)
                total_nll_posterior_pf += float(nll_posterior_pf.item())

                # LSTM
                if diff_lstm_model_path is not None:
                    lstm_predicted_states, lstm_cov = lstm_filter(control.to(torch.float32), measurements.to(torch.float32), inputs[:, 0], diag_cov_prior_batch)
                    total_nll_posterior_lstm += lstm_nll(lstm_predicted_states, lstm_cov, inputs).item()


                if flag_visual_logging and batch_idx == sample_batch:
                    
                    pose_dict = {
                        "GT": inputs[sample_index, traj_idx],
                        # "Diff-HEF": predicted_pose[sample_index],
                        "HEF": predicted_pose_true[sample_index],
                        # "Estimated-HEF": predicted_pose_estimated[sample_index],
                        "EKF": posterior_mean_ekf[sample_index],
                        "HistF": posterior_mean_hist[sample_index], 
                        "PF": posterior_mean_pf[sample_index]
                    }
                    filters_dict = {
                        "GT": [inputs[sample_index, traj_idx].cpu().numpy(), None],
                        "HEF": [predicted_pose_true[sample_index].cpu().numpy(), posterior_true.prob[sample_index].cpu().numpy()],
                        # "Estimated-HEF": [predicted_pose_estimated[sample_index], posterior_estimated.prob[sample_index]],
                        "EKF": [posterior_mean_ekf[sample_index].cpu().numpy(), posterior_cov_ekf[sample_index].cpu().numpy()],
                        "HistF": [posterior_mean_hist[sample_index].cpu().numpy(), posterior_pdf_hist[sample_index].cpu().numpy()],
                        "PF": [posterior_mean_pf[sample_index].cpu().numpy(), posterior_pf[sample_index].cpu().numpy()]
                    }

                    if diff_ekf_model_path is not None:
                        pose_dict["Diff-EKF"] = posterior_mean_diff_ekf[sample_index]
                        filters_dict["Diff-EKF"] = [posterior_mean_diff_ekf[sample_index].cpu().numpy(), posterior_cov_diff_ekf[sample_index].cpu().numpy()]
                    if diff_hist_model_path is not None:
                        pose_dict["Diff-HistF"] = posterior_mean_diff_hist[sample_index]
                        filters_dict["Diff-HistF"] = [posterior_mean_diff_hist[sample_index].cpu().numpy(), posterior_pdf_diff_hist[sample_index].cpu().numpy()]
                    if diff_lstm_model_path is not None:
                        pose_dict["LSTM"] = lstm_predicted_states[sample_index, traj_idx]
                        filters_dict["LSTM"] = [lstm_predicted_states[sample_index, traj_idx].cpu().numpy(), torch.diag(lstm_cov[sample_index, traj_idx]).cpu().numpy()]

                    # visualize_trajectory(
                    #     pose_dict,
                    #     posterior.prob, measurement_model.prob, belief_hat.prob,
                    #     beacons, sample_index, traj_idx, beacon_idx,
                    #     X, Y, T, legend, CONFIG_MEAN_SE2_LF, logging_path,
                    #     f"_diff_hef_{batch_idx}_{sample_index}_{traj_idx}"
                    # )

                    visualize_trajectory(
                        pose_dict,
                        posterior_true.prob.real, measurement_model_true.prob.real, belief_hat_true.prob.real,
                        beacons, sample_index, i, beacon_idx,
                        X, Y, T, legend, CONFIG_MEAN_SE2_LF, logging_path,
                        f"_true_hef_{batch_idx}_{sample_index}_{i}"
                    )
                    titles = ["HEF", "EKF", "HistF", "PF"]
                    # Plot the filters
                    ax_filter = plot_se2_filters(filters_dict,
                                                X.cpu().numpy(),
                                                Y.cpu().numpy(),
                                                T.cpu().numpy(),
                                                beacons,
                                                titles,
                                                config=CONFIG_MEAN_SE2_LF,)
                    for ax in ax_filter:
                        ax.set_xlim(-0.5, 0.5)
                        ax.set_ylim(-0.5, 0.5)
                        ax.scatter(beacons[beacon_idx[sample_index, i], 0].cpu(), beacons[beacon_idx[sample_index, i], 1].cpu(),
                            c='y', marker='o', s=80, alpha=0.8, zorder=2)


                    plt.savefig(logging_path + f"/se2_filter_landmarks_{batch_idx}_{sample_index}_{i}.png")
                    plt.close()

                    # visualize_trajectory(
                    #     pose_dict,
                    #     posterior_estimated.prob.real, measurement_model_estimated.prob.real, belief_hat_estimated.prob.real,
                    #     beacons, sample_index, i, beacon_idx,
                    #     X, Y, T, legend, CONFIG_MEAN_SE2_LF, logging_path,
                    #     f"_estimated_hef_{batch_idx}_{sample_index}_{traj_idx}"
                    # )


            # predicted_trajectory = torch.stack(trajectory_list, dim=1)
            predicted_trajectory_ekf = torch.stack(trajectory_list_ekf, dim=1)
            predicted_trajectory_true = torch.stack(trajectory_list_true, dim=1)
            # predicted_trajectory_estimated = torch.stack(trajectory_list_estimated, dim=1)
            predicted_trajectory_hist = torch.stack(trajectory_list_hist, dim=1)
            
            predicted_trajectory_pf = torch.stack(trajectory_list_pf, dim=1)

            # Save predicted trajectories to a JSON file

            predicted_trajectories = {
                "hef": predicted_trajectory_true.cpu().tolist(),
                "ekf": predicted_trajectory_ekf.cpu().tolist(),
                "hist": predicted_trajectory_hist.cpu().tolist(),
                "pf": predicted_trajectory_pf.cpu().tolist(),
                "gt": inputs.cpu().tolist(),
            }

            predicted_trajectories_path = os.path.join(logging_path, f"trajectories_batch_{batch_idx}.json")
            with open(predicted_trajectories_path, "w") as json_file:
                json.dump(predicted_trajectories, json_file)

            # total_rmse += rmse_se2(inputs, predicted_trajectory)
            total_rmse_true += rmse_se2(inputs, predicted_trajectory_true)
            # total_rmse_estimated += rmse_se2(inputs, predicted_trajectory_estimated)
            total_rmse_ekf += rmse_se2(inputs, predicted_trajectory_ekf)
            total_rmse_hist += rmse_se2(inputs, predicted_trajectory_hist)
            total_rmse_pf += rmse_se2(inputs, predicted_trajectory_pf)

            if diff_ekf_model_path is not None:
                predicted_trajectory_diff_ekf = torch.stack(trajectory_list_diff_ekf, dim=1)
                total_rmse_diff_ekf += rmse_se2(inputs, predicted_trajectory_diff_ekf)
                

            if diff_hist_model_path is not None:
                predicted_trajectory_diff_hist = torch.stack(trajectory_list_diff_hist, dim=1)
                total_rmse_diff_hist += rmse_se2(inputs, predicted_trajectory_diff_hist)
                

        if diff_lstm_model_path is not None:
            total_rmse_lstm += rmse_se2(inputs, lstm_predicted_states[:, 1:, :])
    
        # num_samples = len(test_loader)
        trajectory_length_minus_one = args.trajectory_length - 1

        avg_rmse_true = total_rmse_true / num_samples
        avg_nll_likelihood_true = total_nll_likelihood_true / (num_samples * trajectory_length_minus_one)
        avg_nll_posterior_true = total_nll_posterior_true / (num_samples * trajectory_length_minus_one)

        avg_rmse_ekf = total_rmse_ekf / num_samples
        avg_nll_posterior_ekf = total_nll_posterior_ekf  / (num_samples * trajectory_length_minus_one)

        avg_rmse_diff_ekf = total_rmse_diff_ekf / num_samples
        avg_nll_posterior_diff_ekf = total_nll_posterior_diff_ekf / (num_samples * trajectory_length_minus_one)

        avg_rmse_diff_hist = total_rmse_diff_hist / num_samples
        avg_nll_posterior_diff_hist = total_nll_posterior_diff_hist / (num_samples * trajectory_length_minus_one)

        avg_rmse_hist = total_rmse_hist / num_samples
        avg_nll_posterior_hist = total_nll_posterior_hist / (num_samples * trajectory_length_minus_one)
        
        avg_rmse_pf = total_rmse_pf / num_samples
        avg_nll_posterior_pf = total_nll_posterior_pf / (num_samples * trajectory_length_minus_one)

        avg_rmse_lstm = total_rmse_lstm / num_samples
        avg_nll_posterior_lstm = total_nll_posterior_lstm / (num_samples * trajectory_length_minus_one)
      
        print(f"Average RMSE (HEF): {avg_rmse_true}")
        print(f"Average NLL Likelihood (HEF): {avg_nll_likelihood_true}")
        print(f"Average NLL Posterior (HEF): {avg_nll_posterior_true}")

        print(f"Average RMSE (EKF): {avg_rmse_ekf}")
        print(f"Average NLL Posterior (EKF): {avg_nll_posterior_ekf}")

        print(f"Average RMSE (HistF): {avg_rmse_hist}")
        print(f"Average NLL Posterior (HistF): {avg_nll_posterior_hist}")

        print(f"Average RMSE (PF): {avg_rmse_pf}")
        print(f"Average NLL Posterior (PF): {avg_nll_posterior_pf}")

        print(f"Average RMSE (Diff-HistF): {avg_rmse_diff_hist}")
        print(f"Average NLL Posterior (Diff-HistF): {avg_nll_posterior_diff_hist}")

        print(f"Average RMSE (Diff-EKF): {avg_rmse_diff_ekf}")
        print(f"Average NLL Posterior (Diff-EKF): {avg_nll_posterior_diff_ekf}")

        print(f"Average RMSE (LSTM): {avg_rmse_lstm}")
        print(f"Average NLL Posterior (LSTM): {avg_nll_posterior_lstm}")
        str_run_id = str(wandb.run.id)
        # Log results to wandb as a table
        table = wandb.Table(columns=["Filter", "RMSE", "NLL Likelihood", "NLL Posterior"])
        table.add_data("HEF", str(avg_rmse_true), str(avg_nll_likelihood_true), str(avg_nll_posterior_true))
        table.add_data("EKF", str(avg_rmse_ekf), "N/A", str(avg_nll_posterior_ekf))
        # table.add_data("Diff-HEF", str(avg_rmse), str(avg_nll_likelihood), str(avg_nll_posterior))
        # table.add_data("Estimated HEF", str(avg_rmse_estimated), str(avg_nll_likelihood_estimated), str(avg_nll_posterior_estimated))
        table.add_data("HistF", str(avg_rmse_hist), "N/A", str(avg_nll_posterior_hist))
        table.add_data("PF", str(avg_rmse_pf), "N/A", str(avg_nll_posterior_pf))
        table.add_data("Diff-HistF", str(avg_rmse_diff_hist), "N/A", str(avg_nll_posterior_diff_hist))
        table.add_data("Diff-EKF", str(avg_rmse_diff_ekf), "N/A", str(avg_nll_posterior_diff_ekf))
        table.add_data("LSTM", str(avg_rmse_lstm), "N/A", str(avg_nll_posterior_lstm))

        wandb.log({f"Evaluation Metrics {str_run_id}": table})
        # Log plots to wandb
        for file in os.listdir(logging_path):
            if file.endswith(".png"):
                wandb.log({"Plots": wandb.Image(os.path.join(logging_path, file))})




def parse_list(value):
    """Helper function to parse a list from string input."""
    try:
        return ast.literal_eval(value)  # Safely convert the string into a Python literal (e.g., list)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

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
    
def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_trajectories', type=int, default=10, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=80, help='Length of each trajectory')
    parser.add_argument('--step_motion', type=parse_list, default=[0.01, 0.00, np.pi / 40.0], help='Step motion parameters')
    parser.add_argument('--motion_cov', type=parse_list, default=[0.001, 0.001, 0.001], help='Motion noise parameters')
    parser.add_argument('--measurement_cov', type=float, default=0.0001, help='Measurement noise')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0, help='Test split')
    parser.add_argument('--grid_size', type=parse_list, default=[50, 50, 32], help='Grid size')
    parser.add_argument('--cov_prior', type=parse_list, default=[0.1, 0.1, 0.1], help='Covariance prior')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--decay_rate', type=float, default=5, help='Decay rate for regularization')
    parser.add_argument('--threshold_warmup', type=int, default=200, help='Threshold for warmup')
    parser.add_argument('--learning_rate_start', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=0.0005, help='Final learning rate')
    parser.add_argument('--slope_weight', type=float, default=0.5, help='Slope weight for learning rate decay')
    parser.add_argument('--estimated_measurement_cov', type=float, default=0.0001, help='Estimated measurement noise')
    parser.add_argument('--spline_order', type=int, default=2, help='Order of spline interpolation')
    return parser.parse_args()

if __name__ == "__main__":


    args = parse_args()
    # run = wandb.init(project="Diff-HEF",group="SE2",entity="korra141",
    #           tags=["SE2","DenistyEstimation","UnimodalNoise","Training"],
    #           name="SE2-DiffHEF-RangeSimulator-1",
    #           notes="Diff-HEF on SE2 Range Simulator",
    #           config=args)
    run = wandb.init(mode="disabled",project="Range-Simulator-Inference",group="SE2",entity="korra141",
              tags=["SE2","Inference", "AnalyticalFilters"],
              name="SE2-Range-Simulator",
              config=args)

    run_name = "SE2-Range-Simulator"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # random_number = random.randint(1000, 9999)
    random_number = 2013
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
    # diff_hef_model_path=None
    # diff_ekf_model_path= "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_EKFHEF/20250424_145454_2013/observation_ekf_model_epoch_180.pth"
    # diff_hist_model_path= "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_HistF/20250424_151558_2013/hist_model_epoch_100.pth"
    # diff_ekf_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_EKFHEF/20250501_015421_2013/observation_ekf_model_epoch_399.pth"
    # diff_hist_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_HistF/20250501_020036_2013/hist_model_epoch_399.pth"
    # diff_lstm_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/LSTM_SE2/20250417_001032/lstm_se2_epoch_100.pth"

    # NLP + MSE
    # diff_hef_model_path=None
    # diff_ekf_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_EKFHEF/20250506_003716_2013/observation_ekf_model_epoch_299.pth"
    # # diff_hist_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_HistF/20250501_110409_2013/hist_model_epoch_399.pth"
    # diff_hist_model_path = None
    # diff_lstm_model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/LSTM_SE2/20250505_151616/lstm_se2_epoch_100.pth"

    # Analytical Filters 
    diff_hef_model_path = None
    diff_ekf_model_path = None
    diff_hist_model_path = None
    diff_lstm_model_path = None


    inference(args, logging_path, diff_hef_model_path=diff_hef_model_path,  diff_ekf_model_path=diff_ekf_model_path,  diff_hist_model_path=diff_hist_model_path,  diff_lstm_model_path=diff_lstm_model_path)