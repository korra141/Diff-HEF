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
from src.distributions.R1.HarmonicExponentialDistribution import HarmonicExponentialDistribution as R1_HED
from src.utils.sampler import se2_grid_samples_torch
from src.distributions.SE2.se2_distribution import SE2, SE2Gaussian
from src.filter.bayes_filter import BayesFilter
from src.filter.HistF import BatchedRangeHF
from src.filter.PF import RangePF
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

# device = torch.device("cpu")

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    memory_in_mb = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"[{tag}] Memory Usage: {memory_in_mb:.2f} MB")


# class DensityEstimator(nn.Module):
#     def __init__(self, input_size):
#         super(DensityEstimator, self).__init__()
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

# import torch
# import torch.nn as nn

class DensityEstimator(nn.Module):
    def __init__(self, input_size, measurement_size=1):
        super(DensityEstimator, self).__init__()
        
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


def range_hist_step(inputs, measurements, control, range_beacon, hist_filter, MOTION_NOISE, MEASUREMENT_NOISE):

    batch_size = inputs.shape[0]
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    # measurement_cov = torch.diag(torch.tensor(MEASUREMENT_NOISE ** 2).unsqueeze(0)).to(torch.float64).to(device)
    # measurement_cov_ = torch.tile(measurement_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    hist_filter.prediction(control, motion_model_cov_)
    dist = torch.linalg.norm(range_beacon  - hist_filter.grid_samples[:, :, 0:2], dim=-1)
    energy = torch.distributions.Normal(measurements, MEASUREMENT_NOISE).log_prob(dist)
    # Normalize log likelihoods
    energy -= torch.logsumexp(energy, dim=0)
    posterior_mean, posterior_pdf = hist_filter.update(energy)
    nll_posterior = hist_filter.neg_log_likelihood(inputs, posterior_pdf.view(batch_size, -1))
    return posterior_mean, posterior_pdf , nll_posterior

def train_hist_step(range_beacon, model, hist_filter, inputs, measurements, control, args):
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    motion_model_cov = torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float64).to(device)
    motion_model_cov_ = torch.tile(motion_model_cov.unsqueeze(0), [inputs.shape[0], 1, 1])
    with torch.no_grad():
        # Prediction step
        belief_hat = hist_filter.prediction(control,  motion_model_cov_)

    # Measurement model using the provided model
    measurement_pdf = model(belief_hat.to(torch.float32), measurements.to(torch.float32))
    energy = torch.log(measurement_pdf.to(torch.float64) + 1e-8)
    energy_ = energy - torch.logsumexp(energy, dim=1, keepdim=True)  # Normalize log likelihoods

    # with torch.no_grad():
    posterior_mean_hist, posterior_pdf_hist = hist_filter.update(energy_)

    nll_posterior_hist = hist_filter.neg_log_likelihood(inputs,posterior_pdf_hist.view(args.batch_size, -1) )
    nll_measurement_hist = hist_filter.neg_log_likelihood_measurement(range_beacon, measurements, energy_)

    return posterior_mean_hist, posterior_pdf_hist, nll_posterior_hist, nll_measurement_hist

def training_hist(logging_path, args, model_path=None):
    beacons = torch.tensor(
            [[0, 0.1],
             [0, 0.05],
             [0, 0.0],
             [0, -0.05],
             [0, -0.1]]).to(device)
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

    model = DensityEstimator(math.prod(args.grid_size)).to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        epoch_start = int(model_path.split("_")[-1].split(".")[0])
    else:
        initialize_weights(model)
        epoch_start = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    lr_decay = lambda epoch: (args.learning_rate_end / args.learning_rate_start) ** ((epoch / args.num_epochs) ** args.slope_weight)

    for epoch in range(epoch_start, args.num_epochs):
        model.train()
        total_loss = 0
        total_rmse = 0
        total_nll_posterior = 0
        total_nll_likelihood = 0

        if epoch < args.threshold_warmup:
            regularizer_weight = math.exp(-args.decay_rate * (epoch / args.threshold_warmup))
        else:
            regularizer_weight = math.exp(-args.decay_rate * ((epoch - args.threshold_warmup) / (args.num_epochs - args.threshold_warmup)))

        new_lr = args.learning_rate_start * lr_decay(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        for batch_idx, data in enumerate(train_loader):
            start_time = datetime.datetime.now()
            
            inputs, measurements, control, beacon_idx = data
            inputs, measurements, control, beacon_idx = inputs.to(device), measurements.to(device), control.to(device),beacon_idx.to(device)

            hist_filter = BatchedRangeHF(args.batch_size, inputs[:, 0], cov_prior_batch, poses, X, Y, T, grid_size=args.grid_size, device=device)
            trajectory_list = []
            trajectory_list.append(inputs[:, 0])  # Append the first input as the initial state

            for i in range(args.trajectory_length - 1):
                traj_idx = i + 1
                range_beacon = beacons[beacon_idx[:, i], :]
                posterior_mean, posterior_pdf, nll_posterior, nll_measurement_likelihood = train_hist_step(
                    range_beacon, model, hist_filter, inputs[:, traj_idx], measurements[:, i], control[:, i], args
                )

                # print(F.mse_loss(posterior_mean, inputs[:, traj_idx]))
                # if epoch < args.threshold_warmup:
                # loss = nll_measurement_likelihood.to(torch.float32)
                loss = 0.5*(nll_posterior + mse(posterior_mean, inputs[:, traj_idx])).to(torch.float32)
                # loss = nll_posterior.to(torch.float32)
                #     loss = (regularizer_weight * nll_measurement_likelihood + (1 - regularizer_weight) * F.mse_loss(posterior_mean, inputs[:, traj_idx])).to(torch.float32)
                # loss = nll_posterior.to(torch.float32)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_nll_likelihood += float(nll_measurement_likelihood.item())
                total_nll_posterior += float(nll_posterior.item())
                trajectory_list.append(posterior_mean)
                # print(f"Epoch: {epoch}, batch: {batch_idx}, step: {i}, loss: {loss.item()}, nll_posterior: {nll_posterior.item()}, nll_measurement_likelihood: {nll_measurement_likelihood.item()}, time: {datetime.datetime.now() - start_time}")
            predicted_trajectory = torch.stack(trajectory_list, dim=1)
            total_rmse += rmse_se2(inputs, predicted_trajectory)
            # print(f"Epoch: {epoch}, batch: {batch_idx}, time: {datetime.datetime.now() - start_time}")
        avg_loss = total_loss / (len(train_loader) * (args.trajectory_length - 1))
        avg_rmse = total_rmse / len(train_loader)
        avg_nll_likelihood = total_nll_likelihood / (len(train_loader) * (args.trajectory_length - 1))
        avg_nll_posterior = total_nll_posterior / (len(train_loader) * (args.trajectory_length - 1))

        wandb.log({
            "Training Loss": avg_loss,
            "Training RMSE": avg_rmse,
            "Training NLL Likelihood": avg_nll_likelihood,
            "Training NLL Posterior": avg_nll_posterior,
            "Epoch": epoch,
            "Regularizer Weight": regularizer_weight, 
            "Learning Rate": new_lr,
        })

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, NLL Likelihood: {avg_nll_likelihood:.4f}, NLL Posterior: {avg_nll_posterior:.4f}")

        if (epoch+1) % 100 == 0:
            model_save_path = os.path.join(logging_path, f"hist_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path, base_path=logging_path)

def nll_hist_r1(measurements, energy):
    # energy #[batch_size, N]
    # measurements #[batch_size, 1]
    # energy is over this grid : dist = torch.linalg.norm(range_beacon  - hist_filter.grid_samples[:, :, 0:2], dim=-1) #[batch_size, N] 
    return torch.mean(-energy[measurements.floor().long()])

import ast

def parse_list(value):
    """Helper function to parse a list from string input."""
    try:
        return ast.literal_eval(value)  # Safely convert the string into a Python literal (e.g., list)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

    
def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs')
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
    parser.add_argument('--threshold_warmup', type=int, default=200, help='Threshold for warmup')
    parser.add_argument('--learning_rate_start', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=0.0005, help='Final learning rate')
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
    run = wandb.init(project="Diff-HistF",group="SE2",entity="korra141",
              tags=["SE2","ExtraTraining", "NLLPosterior+MSE"],
              name="SE2-DiffHistF-RangeSimulator-1",
              notes="Diff-HistF on SE2 Range Simulator",
              config=args)
    # artifact = wandb.Artifact("SE2_Range_DiffHEF", type="script")
    # artifact.add_file(__file__)
    # run.log_artifact(artifact)
    # run_name = "SE2_Range_DiffHEF"
    run_name = "SE2_Range_HistF"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_HistF/20250430_190039_2013/hist_model_epoch_199.pth"
    model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/SE2_Range_HistF/20250605_125822_7825/hist_model_epoch_99.pth"
    training_hist(logging_path, args, model_path)
