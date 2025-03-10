import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import math
import datetime
import time
import random
import uuid
import pdb
import psutil
import logging
import logging.handlers
import multiprocessing as mp
import wandb

pid = os.getpid()
base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)

from src.utils.visualisation import plot_density, plot_distributions_s1
from src.distributions.R2.StandardDistribution import GaussianDistribution as GaussianDistribution_r2
from src.distributions.R2.HarmonicExponentialDistribution import HarmonicExponentialDistribution as HarmonicExponentialDistribution_r2
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution as HarmonicExponentialDistribution_s1
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution,VonMissesDistribution_torch
from src.distributions.SE2.SE2_FFT import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from src.data_generation.SE2.uncertainity_estimation import create_dataloaders
from src.utils.metrics import fast_wasserstein_geomloss, kl_divergence_r2, kl_divergence_k, wasserstein_distance_s1,wasserstein_distance_s1_simple
from src.utils.debug_tools import setup_logger
import argparse

def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            # torch.nn.init.zeros_(module.bias)
            torch.nn.init.constant_(module.bias, 0.01)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
        
        if module.bias is not None:
            # torch.nn.init.zeros_(module.bias)
            torch.nn.init.constant_(module.bias, 0.01)


class R2DensityEstimator(nn.Module):
    def __init__(self, grid_size=(50, 50)):
        super(R2DensityEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.Softplus()
        )
        self.input_padding = nn.ReplicationPad2d(1)
        self.grid_size = grid_size
        
    def forward(self, r2_input):
        r2_input = self.input_padding(r2_input)
        r2_density = self.network(r2_input).squeeze(1)  # Shape: [batch_size, H, W]
        r2_density = r2_density[:,1:1+self.grid_size[0], 1:1+self.grid_size[1]]
        return r2_density

class S1EnergyEstimator(nn.Module):
    def __init__(self, grid_size=32):
        super(S1EnergyEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1, 10),  # Input: theta (1D)
            nn.ReLU(),
            nn.Linear(10, grid_size),  # Output: density on S1
        )
        
    def forward(self, s1_input):
        s1_energy = self.network(s1_input)  # Shape: [batch_size, grid_size]
        return s1_energy
        
def validate_model_r2(model_r2, val_loader, epoch, args, device, logging_path, logger, run):
    model_r2.eval()
    hed_r2 = HarmonicExponentialDistribution_r2(args.grid_size[:-1], [0.05, 0.05], range_x=[-0.5, 0.5], range_y=[-0.5, 0.5])
    true_cov_r2 = torch.tile(torch.tensor([args.measurement_noise[0]**2, args.measurement_noise[1]**2]).unsqueeze(0), (args.batch_size, 1))
    validation_path = os.path.join(logging_path, "r2", "validation")
    grid_x = torch.linspace(-0.5, 0.5, args.grid_size[0]).to(device)
    grid_y = torch.linspace(-0.5, 0.5, args.grid_size[1]).to(device)
    os.makedirs(validation_path, exist_ok=True)
    sample_batch = 0
    running_loss = 0.0
    nll_tot = 0.0
    true_nll_tot = 0.0
    smoothness_tot = 0.0
    wl_tot = 0.0
    kl_tot = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            initial_distribution_r2 = GaussianDistribution_r2(inputs[:, 0:2], true_cov_r2.to(device), args.grid_size[0:2], x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])
            initial_density_r2 = initial_distribution_r2.density_over_grid().to(torch.float32)
            output_r2 = model_r2(initial_density_r2.unsqueeze(1))
            z_r2 = hed_r2.normalization_constant(output_r2)
            z_r2_processed = torch.where(z_r2 == 0, torch.tensor(1.0), z_r2)
            output_r2_density = output_r2 / z_r2_processed 
            nll_r2 = torch.mean(hed_r2.loss_energy(output_r2_density, targets[:, 0:2]))

            diff_dim1 = output_r2_density[:, :, 1:] - output_r2_density[:, :, :-1]
            diff_dim2 = output_r2_density[:, 1:, :] - output_r2_density[:, :-1, :]
            smoothness_loss_r2 = torch.mean(diff_dim1 ** 2) + torch.mean(diff_dim2 ** 2)

            target_distribution_r2 = GaussianDistribution_r2(inputs[:, 0:2], true_cov_r2.to(device), args.grid_size[:-1], x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])
            true_density_plot_r2 = target_distribution_r2.density_over_grid()
            true_nll_tot += target_distribution_r2.negative_log_likelihood(targets[:, 0:2])

            wl_tot += fast_wasserstein_geomloss(output_r2_density, true_density_plot_r2)
            kl_tot += kl_divergence_r2(output_r2_density, true_density_plot_r2, grid_x, grid_y)

            loss_r2 = nll_r2 + smoothness_loss_r2
            running_loss += loss_r2.item()
            nll_tot += nll_r2.item()
            smoothness_tot += smoothness_loss_r2.item()

            if epoch % 50 == 0 and batch_idx == sample_batch:
                j = np.random.choice(args.batch_size, 1, replace=False)[0]

                plot_dict = {
                    'true_density': true_density_plot_r2[j],
                    'predicted_density': output_r2_density[j],
                }
                plot_density(inputs[j, 0:2], targets[j, 0:2], [-0.5, 0.5], [-0.5, 0.5], validation_path, plot_dict, f"validation_r2_epoch_{epoch}_batch_{batch_idx}_sample{j}")

    epoch_loss = running_loss / len(val_loader)
    logger.info(f"R2 Validation, Loss: {epoch_loss:.4f}, NLL: {nll_tot / len(val_loader):.4f}, Smoothness Loss: {smoothness_tot / len(val_loader):.4f}, True NLL: {true_nll_tot / len(val_loader):.4f}, Wasserstein Loss: {wl_tot / len(val_loader):.4f}, KL Loss: {kl_tot / len(val_loader):.4f}")
    metrics = {
          'R2 Epoch': epoch + 1,
          'R2 Val NLL' : nll_tot / len(val_loader),
          'R2 Val True NLL': true_nll_tot / len(val_loader),
        #   'R2 Val Smoothness Loss': smoothness_tot / len(val_loader),
          'R2 Val Loss': epoch_loss,
          'R2 Val KL Divergence': kl_tot/ len(val_loader),
          'R2 Val Wasserstein Distance': wl_tot / len(val_loader)}
    run.log(metrics)

def train_model_r2(train_loader,  val_loader, args, device, logging_path, log_queue, run, output_queue):
    # print(f"Training R2 model {rank} on {device}")
    lambda_scheduler = lambda epoch: args.start_lambda * torch.exp(torch.tensor(-args.reg_factor * epoch / args.num_epochs))
    if args.reg_factor == 0:
        lambda_ = 0
    process = psutil.Process(pid)
    mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"Memory Usage: {mem:.2f} GB")
    
    logger = setup_logger(log_queue)
    # logger.info(f"Training started in process {rank}")
    model_r2 = R2DensityEstimator(args.grid_size[:2]).to(device)
    training_path = os.path.join(logging_path, "r2", "training")
    os.makedirs(training_path, exist_ok=True)
    initialize_weights(model_r2)
    optimizer_r2 = optim.Adam(model_r2.parameters(), lr=args.learning_rate)
    hed_r2 = HarmonicExponentialDistribution_r2(args.grid_size[:-1], [0.05, 0.05], range_x=[-0.5, 0.5], range_y=[-0.5, 0.5])
    true_cov_r2 = torch.tile(torch.tensor([args.measurement_noise[0]**2, args.measurement_noise[1]**2]).unsqueeze(0), (args.batch_size, 1))
    grid_x = torch.linspace(-0.5, 0.5, args.grid_size[0]).to(device)
    grid_y = torch.linspace(-0.5, 0.5, args.grid_size[1]).to(device)
    sample_batch = 0
    batch_outputs = []
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        nll_tot = 0.0
        true_nll_tot = 0.0
        smoothness_tot = 0.0
        output_r2_tot = 0.0
        wl_tot = 0.0
        kl_tot = 0.0
        learning_rate_decay = args.learning_rate + (args.learning_rate_end - args.learning_rate) * (1 - np.exp((-args.lr_factor * epoch / args.num_epochs)))
        for param_group in optimizer_r2.param_groups:
            param_group['lr'] = learning_rate_decay
        start_time = time.time()
        validate_model_r2(model_r2, val_loader, epoch, args, device, logging_path, logger, run)
        end_time = time.time()
        if args.reg_factor:
            lambda_ = lambda_scheduler(epoch)
            print(f"Epoch {epoch}, Regularisation constant : {lambda_}")    
        # print(f"Validation completed in {end_time - start_time:.2f} seconds")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            initial_distribution_r2 = GaussianDistribution_r2(inputs[:, 0:2], true_cov_r2.to(device), args.grid_size[0:2], x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])
            initial_density_r2 = initial_distribution_r2.density_over_grid().to(torch.float32)
            initial_density_r2.requires_grad = True
            inputs[:, 2].requires_grad = True

            output_r2 = model_r2(initial_density_r2.unsqueeze(1))
            z_r2 = hed_r2.normalization_constant(output_r2)
            z_r2_processed = torch.where(z_r2 == 0, torch.ones_like(z_r2), z_r2)
            output_r2_density = output_r2 / z_r2_processed
            if torch.isnan(output_r2_density).any():
                print("Output R2 Density has NaN values")
                # pdb.set_trace()
            
            nll_r2 = torch.mean(hed_r2.loss_energy(output_r2_density, targets[:, 0:2]))
            diff_dim1 = output_r2_density[:, :, 1:] - output_r2_density[:, :, :-1]
            diff_dim2 = output_r2_density[:, 1:, :] - output_r2_density[:, :-1, :]
            smoothness_loss_r2 = torch.mean(diff_dim1 ** 2) + torch.mean(diff_dim2 ** 2)
            start_metrics = time.time()
            # True distribution
            target_distribution_r2 = GaussianDistribution_r2(inputs[:,0:2], true_cov_r2.to(device),args.grid_size[:-1],x_range=[-0.5,0.5],y_range=[-0.5,0.5])
            true_density_plot_r2 = target_distribution_r2.density_over_grid()
            true_nll_tot += target_distribution_r2.negative_log_likelihood(targets[:,0:2])
            wl_tot += fast_wasserstein_geomloss(output_r2_density, true_density_plot_r2)
            kl_tot += kl_divergence_r2(output_r2_density, true_density_plot_r2, grid_x, grid_y)
            end_metrics = time.time()
            loss_regularisation = hed_r2.loss_regularisation_norm(lambda_,output_r2_density, targets[:, 0:2])
            
            # print(f"Metrics computation time: {end_metrics - start_metrics:.4f} seconds")
            loss_r2 = loss_regularisation 
            # + smoothness_loss_r2 
            if torch.isnan(loss_r2).any() or loss_r2 == 0:
                print("Loss has NaN values or is zero")
            start_backward = time.time()
            optimizer_r2.zero_grad()
            loss_r2.backward()
            # total_norm = 0.0
            # for param in model_r2.parameters():
            #     if param.grad is not None:  # Check if the gradient is computed
            #         param_norm = param.grad.data.norm(2)  # L2 norm of the gradient
            #         total_norm += param_norm.item() ** 2

            # total_norm = total_norm ** 0.5  # Final L2 norm
            # print("Gradient Norm (L2):", total_norm)
            optimizer_r2.step()
            # pdb.set_trace()
            end_backward = time.time()
            # print(f"Backward pass time: {end_backward - start_backward:.4f} seconds")

            running_loss += loss_r2.item()
            nll_tot += nll_r2.item()
            smoothness_tot += smoothness_loss_r2.item()
            mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            # print(f"Memory Usage: {mem:.2f} GB")

            # Visualisting the distribution
            if epoch % 50 == 0 and batch_idx == sample_batch:
                indices = np.random.choice(args.batch_size, 3, replace=False)
                for j in indices:
                    plot_dict = {}
                    plot_dict = {
                        'true_density': true_density_plot_r2[j],
                        'predicted_density': output_r2_density[j],
                        # torch.sum(output_density[j],dim=2) * (2*math.pi/grid_size[2]),
                        }
                    plot_density(inputs[j,0:2], targets[j,0:2],[-0.5,0.5],[-0.5,0.5],training_path,plot_dict,f"training_r2_epoch_{epoch}_batch_{batch_idx}_sample{j}")
            end_time = time.time()
        batch_outputs.append(nll_tot/len(train_loader))
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"R2 Training Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}, NLL: {nll_tot / len(train_loader):.4f}, Smoothness Loss: {smoothness_tot / len(train_loader):.4f}, True NLL: {true_nll_tot / len(train_loader):.4f}, Wasserstein Loss: {wl_tot / len(train_loader):.4f}, KL Loss: {kl_tot / len(train_loader):.4f}, Time Taken: {time.time() - start_time:.2f} seconds")
        metrics = {
          'R2 Epoch': epoch + 1,
          'R2 Training NLL' : nll_tot / len(train_loader),
          'R2 Training True NLL': true_nll_tot / len(train_loader),
        #   'R2 Training Smoothness Loss': smoothness_tot / len(train_loader),
          'R2 Training Loss': epoch_loss,
          'R2 Training KL Divergence': kl_tot/ len(train_loader),
          'R2 Training Wasserstein Distance': wl_tot / len(train_loader)}
        run.log(metrics)
    model_save_path = os.path.join(training_path, 'model_r2.pth')
    torch.save(model_r2.state_dict(), model_save_path)
    run.save(model_save_path,base_path=training_path)
    output_queue.put((1, batch_outputs))

def validate_model_s1(model_s1, val_loader, epoch, args, device, logging_path, logger, run):
    model_s1.eval()
    hed_s1 = HarmonicExponentialDistribution_s1(args.grid_size[2])
    validation_path = os.path.join(logging_path, "s1", "validation")
    os.makedirs(validation_path, exist_ok=True)
    sample_batch = 0
    running_loss = 0.0
    nll_tot = 0.0
    nll_tot_true = 0.0
    wl_tot = 0.0
    kl_div_tot = 0.0
    grid = torch.linspace(0, 2*math.pi, args.grid_size[2] + 1)[:-1].to(device)
    true_cov = torch.tile(torch.tensor(args.measurement_noise[2]**2), (args.batch_size, 1))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output_s1_energy = model_s1(inputs[:, 2].unsqueeze(1))
            z_s1 = hed_s1.normalization_constant(output_s1_energy)
            nll_s1 = hed_s1.negative_log_likelihood(output_s1_energy, targets[:, 2].unsqueeze(-1) + math.pi)
            output_s1 = torch.exp(output_s1_energy - z_s1)
            diff_dim3 = output_s1[:, 1:] - output_s1[:, :-1]
            smoothness_loss_s1 = torch.mean(diff_dim3 ** 2)
            loss = nll_s1
            running_loss += loss.item()
            nll_tot += nll_s1.item()
            true_distribution = VonMissesDistribution_torch((inputs[:, 2] + math.pi).unsqueeze(-1), true_cov.to(device), args.grid_size[2])
            true_density = true_distribution.density()
            nll_tot_true += true_distribution.negative_loglikelihood((targets[:, 2] + math.pi).unsqueeze(-1))
            wl_tot += wasserstein_distance_s1_simple(output_s1,true_density, grid)
            kl_div_tot += kl_divergence_k(output_s1, true_density, grid)
            if epoch % 50 == 0 and batch_idx == sample_batch:
                j = np.random.choice(args.batch_size, 1, replace=False)[0]
                # for j in indices:
                plot_distributions_s1(args.measurement_noise[2], args.grid_size[2], output_s1_energy[j], inputs[:, 2] + math.pi, targets[:, 2] + math.pi, epoch, batch_idx, j, validation_path)
    epoch_loss = running_loss / len(val_loader)
    logger.info(f"S1 Validation, Loss: {epoch_loss:.4f}, NLL: {nll_tot / len(val_loader):.4f}, True NLL: {nll_tot_true / len(val_loader):.4f}, Wasserstein Loss: {wl_tot / len(val_loader):.4f}, KL Loss: {kl_div_tot / len(val_loader):.4f}")
    metrics = {
          'S1 Epoch': epoch + 1,
          'S1 Val NLL' : nll_tot / len(val_loader),
          'S1 Val True NLL': nll_tot_true / len(val_loader),
          'S1 Val Loss': epoch_loss,
          'S1 Val KL Divergence': kl_div_tot/ len(val_loader),
          'S1 Val Wasserstein Distance': wl_tot / len(val_loader)}
    run.log(metrics)

def train_model_s1( train_loader, val_loader, args, device, logging_path, log_queue, run, output_queue):
    logger = setup_logger(log_queue)
    training_path = os.path.join(logging_path, "s1", "training")
    os.makedirs(training_path, exist_ok=True)
    model_s1 = S1EnergyEstimator(args.grid_size[2]).to(device)
    initialize_weights(model_s1)
    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=args.learning_rate)
    hed_s1 = HarmonicExponentialDistribution_s1(args.grid_size[2])
    grid = torch.linspace(0, 2*math.pi, args.grid_size[2] + 1)[:-1].to(device)
    batch_outputs = []
    sample_batch = 0
    true_cov = torch.tile(torch.tensor(args.measurement_noise[2]**2), (args.batch_size, 1))
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        nll_tot = 0.0
        nll_tot_true = 0.0
        output_s1_tot = 0.0
        wl_tot = 0.0
        kl_div_tot = 0.0
        learning_rate_decay = args.learning_rate + (args.learning_rate_end - args.learning_rate) * (1 - np.exp((-args.lr_factor * epoch / args.num_epochs)))
        start_time = time.time()
        for param_group in optimizer_s1.param_groups:
            param_group['lr'] = learning_rate_decay
        validate_model_s1(model_s1, val_loader, epoch, args, device, logging_path, logger, run)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs[:, 2].requires_grad = True
            output_s1_energy = model_s1(inputs[:, 2].unsqueeze(1))
            z_s1 = hed_s1.normalization_constant(output_s1_energy)
    
            nll_s1 = hed_s1.negative_log_likelihood(output_s1_energy, targets[:,2].unsqueeze(-1) + math.pi)
            output_s1 = torch.exp(output_s1_energy - z_s1)
            
            diff_dim3 = output_s1[:, 1:] - output_s1[:, :-1]
            smoothness_loss_s1 = torch.mean(diff_dim3 ** 2)

            loss = nll_s1 
            optimizer_s1.zero_grad()
            loss.backward()
            optimizer_s1.step()

            running_loss += loss.item()
            nll_tot += nll_s1.item()

            # True distribution 
            true_distribution = VonMissesDistribution_torch((inputs[:, 2] + math.pi).unsqueeze(-1), true_cov.to(device), args.grid_size[2])
            true_density = true_distribution.density()
            nll_tot_true += true_distribution.negative_loglikelihood((targets[:,2] + math.pi).unsqueeze(-1))
            wl_tot += wasserstein_distance_s1_simple(output_s1,true_density, grid)
            kl_div_tot += kl_divergence_k(output_s1, true_density, grid)

            # Visualisting the distribution
            if epoch % 50 == 0 and batch_idx == sample_batch:
                indices = np.random.choice(args.batch_size, 3, replace=False)
                for j in indices:
                    plot_distributions_s1(args.measurement_noise[2], args.grid_size[2], output_s1_energy[j], inputs[:, 2] + math.pi, targets[:, 2] + math.pi, epoch, batch_idx, j, training_path)
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"S1 Training Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}, NLL: {nll_tot / len(train_loader):.4f}, True NLL: {nll_tot_true / len(train_loader):.4f}, Wasserstein Loss: {wl_tot / len(train_loader):.4f}, KL Loss: {kl_div_tot / len(train_loader):.4f}")
        metrics = {
          'S1 Epoch': epoch + 1,
          'S1 Training NLL' : nll_tot / len(train_loader),
          'S1 Training True NLL': nll_tot_true / len(train_loader),
          'S1 Training Loss': epoch_loss,
          'S1 Training KL Divergence': kl_div_tot/ len(train_loader),
          'S1 Training Wasserstein Distance': wl_tot / len(train_loader)}
        run.log(metrics)
        batch_outputs.append(nll_tot/len(train_loader))
    model_save_path = os.path.join(training_path, 'model_s1.pth')
    torch.save(model_s1.state_dict(), model_save_path)
    run.save(model_save_path, base_path=training_path)
    output_queue.put((0, batch_outputs))
def main(args, run):
    train_loader, val_loader = create_dataloaders(args.num_trajectories, args.trajectory_length, args.step_motion, args.motion_noise, args.measurement_noise, args.batch_size, args.validation_split)
    
    run_name = "UE_SE2"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)

    # Set up logging
    log_file = os.path.join(logging_path, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    log_queue = mp.Queue()

    # Start a QueueListener in the main process to handle logs from the queue
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()

    # Log initial information from the main process
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Log initial information
    logger.info("Starting training")
    logger.info(f"Number of trajectories: {args.num_trajectories}")
    logger.info(f"Trajectory length: {args.trajectory_length}")
    logger.info(f"Step motion: {args.step_motion}")
    logger.info(f"Initial Covariance: {args.initial_covariance}")
    logger.info(f"Motion noise: {args.motion_noise}")
    logger.info(f"Measurement noise: {args.measurement_noise}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Validation split: {args.validation_split}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Learning rate end: {args.learning_rate_end}")
    logger.info(f"Learning rate decay factor: {args.lr_factor}")

    true_cov_se2 = torch.tensor([args.measurement_noise[0]**2, args.measurement_noise[1]**2, args.measurement_noise[2]**2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # train_model_r2(train_loader, val_loader, args, device, logging_path,log_queue)
    results = {}
    output_queue = mp.Queue()
    process_1 = mp.Process(target=train_model_r2, args=(train_loader, val_loader, args, device, logging_path, log_queue, run, output_queue))
    process_2 = mp.Process(target=train_model_s1, args=(train_loader, val_loader, args, device, logging_path, log_queue, run, output_queue))
    process_1.daemon = True
    process_2.daemon = True
    process_1.start()
    process_2.start()
    process_1.join()
    process_2.join()

    # Plot the NLL for SE2 by taking the product of R2 x S1
    for _ in range(2):
        rank, output = output_queue.get()
        results[rank] = output 

    
    # True NLL calculation
    cov = torch.diag(true_cov_se2.to(device)).to(torch.float32)
    poses, X, Y, T = se2_grid_samples_torch(args.batch_size ,args.grid_size)
    true_nll_tot_gaussian = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        true_distribution_se2 = GaussianDistribution_se2(inputs.unsqueeze(1), cov,args.grid_size)
        nll_se2_gaussian = true_distribution_se2.negative_log_likelihood(targets.unsqueeze(1)).reshape((args.batch_size, -1))
        true_nll_tot_gaussian += nll_se2_gaussian.mean().item()

    logger.info(f"True NLL Gaussian: {true_nll_tot_gaussian / len(train_loader):.4f}")

    combined_density = []
    for i, items in enumerate(zip(*results.values())):
        result = items[0]  + items[1] 
        combined_density.append(result)  # Independent product of the densities
        logger.info(f"Epoch {i + 1}, NLL: {result:.4f}")
        run.log({"SE2 NLL": result, "Epoch": i, "True SE2 NLL Gaussian": true_nll_tot_gaussian / len(train_loader)})


    print("Logging images to wandb")
    start_wandb_time = time.time()
    folders = ['s1', 'r2']
    subfolders = ['validation', 'training']
    # Iterate through s1/r2 and their subfolders
    for folder in folders:
        for subfolder in subfolders:
            folder_path = os.path.join(logging_path, folder, subfolder)
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.endswith(".png"):
                        run.log({f"{folder}_{subfolder}_{img_file}": wandb.Image(os.path.join(folder_path, img_file))})
    print(f"Logging images to wandb took {time.time() - start_wandb_time} seconds")
    listener.stop()

def parse_args():
    parser = argparse.ArgumentParser(description="Train SE2 Uncertainty Estimation Model")
    parser.add_argument('--num_trajectories', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--learning_rate_end', type=float, default=1e-5, help='Final learning rate')
    parser.add_argument('--lr_factor', type=float, default=10, help='Learning rate decay factor')
    parser.add_argument('--validation_split', type=float, default=0.002, help='Validation split ratio')
    parser.add_argument('--step_motion', type=float, nargs=3, default=[0.05, 0.05, 0.005], help='Step motion parameters')
    parser.add_argument('--motion_noise', type=float, nargs=3, default=[0.1, 0.1, 0.1], help='Motion noise parameters')
    parser.add_argument('--measurement_noise', type=float, nargs=3, default=[0.2, 0.2, 0.2], help='Measurement noise parameters')
    parser.add_argument('--initial_covariance', type=float, nargs=3, default=[0.01, 0.01, 0.01], help='Initial covariance parameters')
    parser.add_argument('--grid_size', type=int, nargs=3, default=[50, 50, 32], help='Grid size for the density estimators')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--start_lambda', type=float, default=100, help='Starting value of lambda for regularization')
    parser.add_argument('--reg_factor', type=float, default=10, help='Regularization factor for lambda scheduler')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    run_id = str(uuid.uuid4())

    run = wandb.init(project="Diff-HEF",group="SE2",entity="korra141",
              tags=["SE2","DenistyEstimation","UnimodalNoise","IndependentSpace", "Trial"],
              name="SE2-DensityEstimation",
              config=args, id=run_id)
    artifact = wandb.Artifact("UE_SE2_script_1", type="script")
    artifact.add_file(__file__)
    run.log_artifact(artifact)
    mp.set_start_method('spawn')
    main(args, run)