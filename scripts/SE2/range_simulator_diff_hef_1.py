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
from torch.profiler import profile, record_function, ProfilerActivity
import cProfile
import pstats
from pstats import SortKey
import gc
from memory_profiler import profile

torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on Ampere GPUs
torch.backends.cudnn.benchmark = True  # Optimize CUDA kernels

base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)
pid = os.getpid()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.R1.HarmonicExponentialDistribution import HarmonicExponentialDistribution as R1_HED
from src.distributions.SE2.SE2_torch import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from src.filter.HEF_SE2 import HEFilter
from src.distributions.SE2.se2_distribution import SE2, SE2Gaussian
from src.filter.bayes_filter import BayesFilter
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean, mse
from src.utils.visualisation import plot_se2_mean_filters,plot_se2_filters
import argparse
from torch.utils.data import Dataset
from torch.profiler import profile, record_function, ProfilerActivity

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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

    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2

        print(f"[{tag}] CUDA Memory | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak Allocated: {max_allocated:.2f} MB | Peak Reserved: {max_reserved:.2f} MB")

    print(f"[{tag}] Memory Usage: {memory_in_mb:.2f} MB")

class MeasurementModel(nn.Module):
    def __init__(self, grid_size=(25, 25, 32)):
        super(MeasurementModel, self).__init__()
        
        # Store grid size for later use
        self.grid_size = grid_size
        
        # 3D Convolutional layers to process p(x) and z
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)  # Output layer
        self.input_padding = nn.ReplicationPad3d(1)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)  # Softmax over the last dimension for conditional probability
        
    def forward(self, p_x, z):
        # p_x has shape [batch_size, H, W, D] and z has shape [batch_size, 1]
        
        # Expand z to match the grid size
        z = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape [batch_size, 1, 1, 1, 1]
        z = z.expand(-1, -1, *self.grid_size)  # Now z has shape [batch_size, 1, H, W, D]
        #p_x_padded = self.input_padding(p_x.unsqueeze(1))
        # Concatenate p_x and z along the channel dimension (axis 1)
        x = torch.cat([p_x.unsqueeze(1), z], dim=1)  # Shape [batch_size, 2, H, W, D]
        x_padded = self.input_padding(x)
        
        # Pass through the network
        out = F.leaky_relu(self.conv1(x_padded))
        out = F.leaky_relu(self.conv2(out))
        out = self.relu(self.conv3(out)).squeeze(1)  # Remove the channel dimension
        # x = self.relu(self.conv4(x))
        
        # Apply softmax to get the conditional probability distribution over the grid
        # Here, the softmax is applied across the last dimension (the grid values at each point)
        # x = self.softmax(x.view(x.size(0), -1, x.size(-1)))  # Flatten over the last dimension
        
        return out[:, 1:1+self.grid_size[0], 1:1+self.grid_size[1], 1:1+self.grid_size[2]]
class DensityEstimator(nn.Module):
    def __init__(self, grid_size=(50, 50, 32)):
        super(DensityEstimator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softplus = nn.Softplus()
        self.input_padding = nn.ReplicationPad3d(1)
        self.grid_size = grid_size

    def forward(self, x):
        x = self.input_padding(x)
        x = self.leaky_relu(self.conv1(x))
        # x = self.leaky_relu(self.conv2(x))
        x = self.conv4(x)
        x = self.softplus(x)
        x = x.squeeze(1)
        x = x[:, 1:1+self.grid_size[0], 1:1+self.grid_size[1], 1:1+self.grid_size[2]]
        return x

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

def normalize_angle(input):
    output = input.clone()
    output[:, 2] = (output[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return output

def diff_hef_step(prior, motion_inv_cov, inputs, measurements, range_beacon, control, poses, X, Y, T, grid_size, hef_filter, model, fft, hed_r1, motion_cov, batch_size):
    start_time = datetime.datetime.now()
    with torch.no_grad():
        # print_memory_usage(f"Before HEF Step")
        motion_model = SE2Gaussian(control, motion_cov, motion_inv_cov, samples=poses, fft=fft)
        # print_memory_usage(f"line 155")
        motion_model.normalize()
        # print_memory_usage(f"line 157")
        belief_hat = hef_filter.prediction(prior, motion_model)
        # print_memory_usage(f"After HEF Prediction")

    #measurement_pdf = model(belief_hat.prob.unsqueeze(1).to(torch.float32).to(device))
    measurement_pdf = model(belief_hat.prob.to(torch.float32).to(device), measurements.to(torch.float32))
    # print_memory_usage(f"After Measurement Model Forward Pass")
    measurement_model = SE2(samples=poses,fft=fft)
    # print_memory_usage('line 166')
    energy = torch.log(measurement_pdf.to(torch.float64) + 1e-8)
    # print_memory_usage('line 168')
    measurement_model.eta = measurement_model.fft.analyze(energy)
    # print_memory_usage('line 170')
    measurement_model.energy = measurement_model.fft.synthesize(measurement_model.eta)
    # print_memory_usage('line 172')
    measurement_model.normalize()
    # print_memory_usage(f"After Measurement Model Normalization")
    # predicted_measurement_density_flat = measurement_model.prob.reshape(batch_size, -1)
    
    posterior = hef_filter.update(belief_hat, measurement_model)
    # print_memory_usage(f"After HEF Update")
    nll_posterior = torch.mean(posterior.fft.neg_log_likelihood(posterior.energy, inputs))
    # print_memory_usage(f"After HEF Negative Log Likelihood Calculation")
    with torch.no_grad():
        predicted_pose = compute_weighted_mean(posterior.prob, poses, X, Y, T)
        predicted_pose[..., 2] = (predicted_pose[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        dist = torch.linalg.norm(range_beacon  - poses[:, :, 0:2], dim=-1)
        range_L = dist.max(dim=-1).values - dist.min(dim=-1).values
        range_L = range_L.view(-1, 1, 1)   
        measurements_norm = (measurements - dist.min(dim=-1).values.view(-1,1))
        measurement_energy = measurement_model.energy.reshape(batch_size, -1)
        nll_measurement_likelihood = hed_r1.negative_log_likelihood(measurement_energy, measurements_norm, range_L)
    # print_memory_usage(f"After HEF measurement nll completion")
    return posterior, measurement_model , belief_hat, predicted_pose, nll_measurement_likelihood, nll_posterior

def analytic_hef(inputs, measurements, range_beacon, control, poses, X, Y, T,  grid_size, batch_size, hef_filter, hed_r1, fft, MOTION_NOISE, MEASUREMENT_NOISE):

    motion_model = SE2Gaussian(control, torch.diag(torch.tensor(args.motion_cov)).to(torch.float64).to(device), samples=poses, fft=fft)
    motion_model.normalize()

    belief_hat = hef_filter.prediction(motion_model)

    dist = torch.linalg.norm(range_beacon  - poses[:, :, 0:2], dim=-1)
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
    predicted_pose[..., 2] = (predicted_pose[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi

    dist = torch.linalg.norm(range_beacon  - poses[:, :, 0:2], dim=-1)
    range_L = dist.max(dim=-1).values - dist.min(dim=-1).values
    range_L = range_L.view(-1, 1, 1)
    measurement_energy = measurement_model.energy.reshape(batch_size, -1)
    nll_measurement_likelihood = hed_r1.negative_log_likelihood(measurement_energy, measurements, range_L)

    return posterior, measurement_model , belief_hat,  predicted_pose, nll_measurement_likelihood, nll_posterior

def detach_tensors(obj):
    """
    Recursively detach all tensors inside a Python object (dict, list, tuple, custom objects).
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_tensors(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_tensors(v) for v in obj)
    elif hasattr(obj, '__dict__'):  # If it's a custom object
        # Create a shallow copy first
        new_obj = obj.__class__.__new__(obj.__class__)
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, detach_tensors(v))
        return new_obj
    else:
        return obj
    
def validate(model, epoch, val_loader, poses, X, Y, T, diff_fft, hed_r1, device, args):
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_nll_posterior = 0
    total_nll_likelihood = 0
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    motion_cov = torch.diag(torch.tensor(args.motion_cov)).to(torch.float64).to(device)
    motion_inv_cov = torch.inverse(motion_cov).to(device)
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
            cov_prior = torch.diag(torch.tensor(args.cov_prior)).to(device).to(torch.float64)
            inv_cov_prior = torch.inverse(cov_prior).to(device)
            prior_diff = SE2Gaussian(inputs[:, 0], cov_prior, inv_cov_prior, samples=poses, fft=diff_fft)
            prior_diff.normalize()
            trajectory_list = []
            trajectory_list.append(inputs[:, 0])  # Append the initial pose
            diff_hef_filter = BayesFilter(distribution=SE2, prior=prior_diff, device=device)

            for i in range(args.trajectory_length - 1):
                traj_idx = i + 1
                range_beacon = beacons[beacon_id[:, i], :]
                posterior, measurement_model, belief_hat, predicted_pose, nll_measurement_likelihood, nll_posterior = diff_hef_step(
                    prior_diff, motion_inv_cov, inputs[:, traj_idx], measurements[:, i], range_beacon, control[:, i], poses, X, Y, T, args.grid_size, diff_hef_filter, model, diff_fft, hed_r1, motion_cov, args.batch_size
                )
                prior_diff =  detach_tensors(posterior)
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
    inv_cov_prior = torch.inverse(cov_prior).to(device)
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
    motion_cov = torch.diag(torch.tensor(args.motion_cov)).to(torch.float64).to(device)
    motion_inv_cov = torch.inverse(motion_cov).to(device)
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
    #model = DensityEstimator(grid_size).to(device)
    model = MeasurementModel(grid_size).to(device)
    initialize_weights(model)
    hed_r1 = R1_HED(math.prod(grid_size), torch.sqrt(torch.tensor(2)))
    diff_fft = SE2_FFT(spatial_grid_size=grid_size,
                    device = device,
                    interpolation_method='spline',
                    spline_order=args.spline_order,
                    oversampling_factor=3)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    lr_decay = lambda epoch : (args.learning_rate_end / args.learning_rate_start) ** ((epoch / NUM_EPOCHS) ** args.slope_weight)
    
    epoch = 0
    for epoch in range(NUM_EPOCHS):
        loss_tot = 0
        mean_rmse_tot = 0
        mean_rmse_true_tot = 0
        nll_posterior_tot = 0
        nll_posterior_true_tot = 0
        nll_likelihood_tot = 0
        start_time = datetime.datetime.now()
        # Adjust this value to control the decay speed
        regularizer_weight = math.exp(-decay_rate * ((epoch - args.threshold_warmup) / (NUM_EPOCHS - args.threshold_warmup)))
        # regularizer_weight = 0
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
            # input_normalise = normalize_angle(inputs)
            prior_diff = SE2Gaussian(inputs[:, 0], cov_prior, inv_cov_prior, samples=poses, fft=diff_fft)
            prior_diff.normalize()
            diff_hef_filter = BayesFilter(distribution=SE2, prior=prior_diff, device=device)
            trajectory_list = []
            trajectory_list.append(inputs[:, 0])  # Append the initial pose
            trajectory_list_true = []
            trajectory_list_true.append(inputs[:, 0])  # Append the initial pose
            sample_idx = random.randint(0, batch_size - 1)
            # Perform operations on inputs and labels using HEF analytical filter here
            for i in range(TRAJECTORY_LENGTH - 1):
                start_time_traj = datetime.datetime.now()
                traj_idx = i + 1
                start_time_step_ = datetime.datetime.now()
                range_beacon = beacons[beacon_id[:, i], :]
                posterior, measurement_model, belief_hat, predicted_pose, nll_measurement_likelihood, nll_posterior = diff_hef_step(prior_diff, motion_inv_cov, inputs[:, traj_idx], measurements[:, i],range_beacon, control[:, i], poses, X, Y, T, grid_size, diff_hef_filter, model, diff_fft, hed_r1, motion_cov, batch_size)
                prior_diff = detach_tensors(posterior)
                # loss = ((1-regularizer_weight) * nll_measurement_likelihood + regularizer_weight * nll_posterior).to(torch.float32)
                # if epoch < args.threshold_warmup:
                # loss = 0.5*(nll_posterior.to(torch.float32) + mse(predicted_pose, inputs[:, traj_idx]).to(torch.float32))
                # print_memory_usage(f"After HEF Step {i} Epoch {epoch} Batch {j}")
                loss = nll_posterior.to(torch.float32)
                # else:
                #     loss = (regularizer_weight * nll_measurement_likelihood + (1 - regularizer_weight) * F.mse_loss(predicted_pose, inputs[:, traj_idx])).to(torch.float32)
                # loss = (regularizer_weight * nll_measurement_likelihood + (1-regularizer_weight) * nll_posterior).to(torch.float32)
                # print_memory_usage("390")
                optimizer.zero_grad(set_to_none=True)
                # print_memory_usage("392")
                loss.backward()
                gc.collect()
                torch.cuda.empty_cache()
                # print_memory_usage("394")
                optimizer.step()
                # print_memory_usage(f"After HEF Loss Backward Step {i} Epoch {epoch} Batch {j}")
                with torch.no_grad():
                    trajectory_list.append(predicted_pose)
                    loss_tot += float(loss.item())
                    nll_likelihood_tot += float(nll_measurement_likelihood.item())
                    nll_posterior_tot += float(nll_posterior.item())
                # print_memory_usage(f"After HEF Step {i} Epoch {epoch} Batch {j} concluded")
                # # Visualise ther results 
                # if epoch % 20 == 0 and j == sample_batch:
                #     epoch_logging_path = os.path.join(logging_path, f"epoch_{epoch}")
                #     os.makedirs(epoch_logging_path, exist_ok=True)
                    
                #     pose_dict = {
                #         "GT": inputs[sample_idx, traj_idx],
                #         "HEF": predicted_pose[sample_idx],
                #     }
                #     axes_mean = plot_se2_mean_filters(
                #                     [belief_hat.prob[sample_idx], measurement_model.prob[sample_idx].detach(), posterior.prob[sample_idx].detach()],
                #                     X,Y,T,
                #                     samples=pose_dict, iteration=i, beacons=beacons[:, :2],
                #                     level_contours=False, contour_titles=legend, config=CONFIG_MEAN_SE2_LF)
                #     for ax_mean in axes_mean:
                #         ax_mean.set_xlim(-0.5, 0.5)
                #         ax_mean.set_ylim(-0.5, 0.5)
                #     axes_mean[3].scatter(beacons[beacon_id[sample_idx, i], 0].cpu(), beacons[beacon_id[sample_idx, i], 1].cpu(),
                #               c='y', marker='o', s=80, alpha=0.8, zorder=2)
                #     plt.savefig(epoch_logging_path + f"/se2_diff_hef_train_landmarks_{epoch}_{j}_{i}.png")
                #     plt.close()
                # # print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {j + 1}/{len(train_loader)}, Step {i + 1}/{TRAJECTORY_LENGTH - 1}, Time: {datetime.datetime.now() - start_time_traj}")
                del posterior, measurement_model, belief_hat
            predicted_trajectory = torch.stack(trajectory_list, dim=1)
            
            with torch.no_grad():
                mean_rmse_tot += rmse_se2(inputs, predicted_trajectory)
                # mean_rmse_true_tot += rmse_se2(inputs, predicted_trajectory_true)

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {j + 1}/{len(train_loader)}, Time: {datetime.datetime.now() - start_time_step}")

            # Create a table for logging
            end_time = datetime.datetime.now()
            len_trajectory_length_adjusted = TRAJECTORY_LENGTH - 1
            table_data = [
                ["Epoch", epoch],
                ["Time", str(end_time - start_time)],
                ["Loss", loss_tot / (len_trajectory_length_adjusted * len(train_loader))],
                ["Diff HEF NLL Likelihood", nll_likelihood_tot / (len_trajectory_length_adjusted * len(train_loader))],
                ["Diff HEF NLL Posterior", nll_posterior_tot / (len_trajectory_length_adjusted * len(train_loader))],
                ["Diff HEF RMSE", mean_rmse_tot / len(train_loader)],
            ]
            wandb.log({ "Epoch": epoch,
                    "Loss": loss_tot / (len_trajectory_length_adjusted * len(train_loader)),
                    "Diff HEF NLL Likelihood": nll_likelihood_tot / (len_trajectory_length_adjusted * len(train_loader)), 
                    "Diff HEF NLL Posterior": nll_posterior_tot / (len_trajectory_length_adjusted * len(train_loader)), 
                    # "True HEF NLL Posterior": nll_posterior_true_tot / (TRAJECTORY_LENGTH * len_train_loader_adjusted), 
                    "Diff HEF RMSE": mean_rmse_tot / len(train_loader),
                    "Regularizer Weight": regularizer_weight,
                    "Learning Rate": new_lr,
                    # "True HEF RMSE": mean_rmse_true_tot / len_train_loader_adjusted 
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

def profile_training(args, logging_path):
    profiler = cProfile.Profile()
    profiler.enable()
    main(args, logging_path)
    profiler.disable()
    log_file = os.path.join(logging_path, "profile_stats.log")
    with open(log_file, "w") as f:
        stats = pstats.Stats(profiler,stream=f).sort_stats(SortKey.TIME)
        stats.print_stats()
    print(f"Profiling stats saved to {log_file}")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_trajectories', type=int, default=300, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=80, help='Length of each trajectory')
    parser.add_argument('--step_motion', type=parse_list, default=[0.01, 0.01, np.pi/40], help='Step motion parameters')
    parser.add_argument('--motion_cov', type=parse_list, default=[0.001, 0.001, 0.001], help='Motion noise parameters')
    parser.add_argument('--measurement_cov', type=float, default=0.0001, help='Measurement noise')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.12, help='Validastion split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--grid_size', type=parse_list, default=[50, 50, 32], help='Grid size')
    parser.add_argument('--cov_prior', type=parse_list, default=[0.1, 0.1, 0.1], help='Covariance prior')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--decay_rate', type=float, default=5, help='Decay rate for regularization')
    parser.add_argument('--threshold_warmup', type=int, default=0, help='Threshold for warmup')
    parser.add_argument('--learning_rate_start', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Final learning rate')
    parser.add_argument('--slope_weight', type=float, default=0.5, help='Slope weight for learning rate decay')
    parser.add_argument('--spline_order', type=int, default=2, help='Order of spline interpolation')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    run = wandb.init(project="Diff-HEF",group="SE2",entity="korra141",
              tags=["SE2","Training", "NLLPosterior", "LearningRateDecrease", "PredictedBelief+Measurement"],
              name="SE2-DiffHEF-RangeSimulator-1",
              notes="Diff-HEF on SE2 Range Simulator",
              config=args)
    # run = wandb.init(mode="disabled", project="Diff-EKF",group="SE2",entity="korra141",
    #           tags=["SE2","DenistyEstimation","UnimodalNoise","Training"],
    #           name="SE2-DiffEKF-RangeSimulator-1",
    #           notes="Diff-EKF on SE2 Range Simulator",
    #           config=args)
    # artifact = wandb.Artifact("SE2_Range_DiffHEF", type="script")
    # artifact.add_file(__file__)
    # run.log_artifact(artifact)
    run_name = "SE2_Range_DiffHEF"
    # run_name = "SE2_Range_EKFHEF"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
        # with record_function("your_model_run"):
    main(args,logging_path)
    # profile_training(args,logging_path)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    # profile_training(args,logging_path)
    # training_ekf(logging_path, args)
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_DiffHEF/20250314_163823_7387/measurement_model_epoch_50.pth"
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_DiffHEF/20250316_231533_3842/measurement_model_epoch_200.pth"
    # inference_old_hef(model_path, args, logging_path)
    # inference(args, logging_path, diff_model_path=None)
