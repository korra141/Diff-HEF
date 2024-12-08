"""
This file contains an experiment to test whether its possible to learn the multimodal measueremnt distribution in S1 using
the Harmonic Exponential Functions. There is no process model and the measurement distribution is learned using 
the true position as input. 

Caveats:
- Since there are multiple trajectories that overlap each other, there is additional ambiguity about which distribution
the measurements are coming from. This is expected to increase the variance of the learned distribution or in the best case
scenario lead to multiple modes in the learned distribution.

The experiment can be run for local grid as well by setting the range_theta parameter.
"""


import time
import torch
import argparse
import torch.fft
import torch.nn as nn
import torch.optim as optim
from scipy.special import i0
import torch.nn.functional as F

import math
import os   

import matplotlib.pyplot as plt
import numpy as np
from time import strftime, localtime
import wandb
import sys
import pdb
import random

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.data_generation.S1.toy_dataset import generating_data_S1_multimodal
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.S1.WrappedNormalDitribution import MultimodalGaussianDistribution,MultimodalGaussianDistribution_torch
from src.utils.visualisation import plot_circular_distribution
from src.utils.metrics import kl_divergence_s1,root_mean_square_error_s1,mean_absolute_error, expected_calibration_error_continuous, compute_cdf_from_pdf, predicted_residual_error, wasserstein_distance

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment to learn the measurement distribution in S1 using Harmonic Exponential Functions.')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=10, help='Number of neurons in the hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--band_limit', type=int, default=100, help='Band limit')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--measurement_noise', type=float, default=0.1, help='Standard deviation of the measurement noise')
    # parser.add_argument('--measurement_noise_max', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument("--mean_offset", type=float, default=np.pi/2, help="Mean offset for the multimodal noise")
    parser.add_argument("--n_modes", type=int, default=2, help="Number of modes for the multimodal noise")
    parser.add_argument("--n_modes_estimated", type=int, default=2, help="Number of modes to be estimated by the model")
    parser.add_argument('--range_theta', type=str, default="None", help='Range of theta for local grid')
    parser.add_argument('--gaussian_parameterisation', type=int, default=1, help='Use Gaussian parameterisation')
    parser.add_argument('--gaussianNLL', type=int, default=1, help='Use Gaussian Negative Log Likelihood')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weights', type=str, default=[0.5,0.5], help='Path to the model weights')
    
    return parser.parse_args()  
def positive_angle(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

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
class SimpleModel(nn.Module):
    def __init__(self,n_modes):
        super(SimpleModel, self).__init__()
        self.n_modes = n_modes
        self.model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3*n_modes)
        )
        self._initialize_weights()

    def forward(self, x):
        y_pred = self.model(x)
        mu = y_pred[:,0:self.n_modes] %(2 * math.pi)
        logcov = y_pred[:,self.n_modes:2*self.n_modes]
        pi = F.softmax(y_pred[:,2*self.n_modes:],dim=1)
        
        epsilon = torch.tensor(-1e-2)  # Small value to avoid zero covariance
        logcov = torch.where(logcov < epsilon, epsilon, logcov)
        return mu, logcov, pi

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
def validate(model, val_loader, args, hed,centers,measurement_noise,logging_path,epoch):
  model.eval()
  val_loss_tot = 0
  val_kl_div_tot = 0
  val_rmse_mu_tot = 0
  val_rmse_cov = 0
  val_mae_mu = 0
  ece_tot = 0
  wd_tot = 0
  log_freq = len(val_loader)/3
  with torch.no_grad():
    for i, (ground_truth, measurements) in enumerate(val_loader):
      if args.gaussian_parameterisation:
        mu, log_cov, pi = model(ground_truth)
        cov = torch.exp(log_cov)
        predicted_distribution = MultimodalGaussianDistribution_torch(mu, cov, pi, args.n_modes_estimated, args.band_limit)
        energy = predicted_distribution.energy()
        predicted_density = predicted_distribution.density()
        mu_ = mu.detach()
        cov_ = cov.detach()
        error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
       
        measurement_cov = torch.ones_like(cov) * (args.measurement_noise **2)
        val_rmse_mu = root_mean_square_error_s1(mu_, ground_truth)
        val_rmse_mu_tot += val_rmse_mu
        val_rmse_cov += root_mean_square_error_s1(cov_, measurement_cov)
        val_mae_mu += mean_absolute_error(mu_, ground_truth)
      else:
        energy = model(ground_truth)
        predicted_density = torch.exp(energy - hed.normalization_constant(energy))
        mode = hed.mode(predicted_density,ground_truth)
        error, hist, bin_centers, bin_edges = predicted_residual_error(mode, ground_truth,20)
        val_rmse_mu = root_mean_square_error_s1(mode, ground_truth)
        val_rmse_mu_tot += val_rmse_mu
        val_mae_mu += mean_absolute_error(mode, ground_truth)
      
      if args.gaussianNLL == 0:
        if args.range_theta is None:
          loss = hed.negative_log_likelihood(energy, measurements)
        else:
          loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
        
      elif args.gaussian_parameterisation and args.gaussianNLL:
        loss = predicted_distribution.negative_loglikelihood(measurements)
      else:
        raise ValueError("Invalid combination of parameters")
      
      mm_mean = positive_angle(ground_truth.unsqueeze(1) + centers)
      true_distribution = MultimodalGaussianDistribution(mm_mean, measurement_noise, args.weights, args.n_modes, args.band_limit)
      true_distribution_energy = true_distribution.energy()
      if args.range_theta is None:
        true_density = true_distribution.density()
      else:
        true_density = true_distribution.density_local(args.range_theta)
      
      val_kl_div_tot += kl_divergence_s1(true_density, predicted_density)
      cdf_predicted_density = compute_cdf_from_pdf(predicted_density)
      cdf_true_density = compute_cdf_from_pdf(true_density)
      ece_tot += expected_calibration_error_continuous(cdf_true_density,cdf_predicted_density,20)
      wd_tot += wasserstein_distance(predicted_density, true_density)
      
      val_loss_tot += loss.item()
      if epoch % 10 == 0 and i % log_freq == 0:
        indices = np.random.choice(args.batch_size, 5, replace=False)
        for j in indices:
          fig, ax = plt.subplots()

          if args.range_theta == None:
            ax = plot_circular_distribution(energy[j],legend="predicted distribution",ax=ax)
          else:
            ax = plot_circular_distribution(energy[j],legend="predicted distribution",mean=ground_truth[j,0],range_theta=args.range_theta,ax=ax)
          ax = plot_circular_distribution(true_distribution_energy[j],legend="true distribution",ax=ax)
          ax.plot(torch.cos(measurements[j]),torch.sin(measurements[j]),'o',label="measurement data")
          ax.plot(torch.cos(ground_truth[j]), torch.sin(ground_truth[j]), 'o', label="pose data")
          ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
          ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
          ax.set_aspect('equal')
          plt.savefig(os.path.join(logging_path, f"validation_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
          # plt.show()
          plt.close()
        plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
        plt.axvline(x=val_rmse_mu, color='r', linestyle='-', label='RMSE')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram predicted residual error')
        plt.savefig(os.path.join(logging_path, f"validation_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
        # plt.show()
        plt.close()

    val_metrics = {
      'Epoch': epoch,
      'Validation NLL Loss': val_loss_tot / len(val_loader),
      'Validation KL Divergence': val_kl_div_tot / len(val_loader),
      'Validation RMSE Mu': val_rmse_mu_tot / len(val_loader),
      'Validation MAE Mu' : val_mae_mu / len(val_loader),
      'Validation ECE': ece_tot / len(val_loader),
      'Validation Wasserstein Distance': wd_tot / len(val_loader)}
    if args.gaussian_parameterisation:
        val_metrics['Validation RMSE Cov'] = val_rmse_cov / len(val_loader)
    
    wandb.log(val_metrics)


def main(args):
    
    measurement_noise = torch.ones(args.n_modes) * args.measurement_noise
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_multimodal(measurement_noise, args.mean_offset,args.n_modes, data_path, args.batch_size, args.n_samples, args.trajectory_length,  args.step_size, True)
    val_loader =  generating_data_S1_multimodal(measurement_noise, args.mean_offset,args.n_modes, data_path, args.batch_size, args.batch_size, args.trajectory_length,  args.step_size, True)

    if args.gaussian_parameterisation:
        model = SimpleModel(args.n_modes_estimated)
    else:
      model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    centers = torch.tile(torch.linspace(-args.mean_offset / 2, args.mean_offset / 2, args.n_modes)[None,:,None], (args.batch_size, 1,1)) # n_modes
    
    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    random_number = random.randint(1000, 9999)
    if args.range_theta == None and args.gaussian_parameterisation == 0:
        run_name = "S1-UE-MM"    
    elif args.range_theta == None and args.gaussian_parameterisation == 1:
        run_name = "S1-UE-MM-Gaussian"
    else:
        run_name = "S1-UE-MM-Local"
    logging_path = os.path.join(base_path,"logs", run_name, str(ctime) + "_" + str(random_number))
    print(f"Logging path: {logging_path}")
    os.makedirs(logging_path)
    # Loss function
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)

    log_freq = len(train_loader)/3
    for epoch in range(args.num_epochs):
        loss_tot = 0
        kl_div_tot = 0
        rmse_mu_tot = 0
        rmse_cov = 0
        mae_mu = 0
        ece_tot = 0
        wd_tot = 0
        validate(model, val_loader, args, hed, centers,measurement_noise,logging_path,epoch)
        for i, (ground_truth, measurements) in enumerate(train_loader):
            start_time = time.time()
            # Forward pass
            if args.gaussian_parameterisation:
              mu, log_cov, pi = model(ground_truth)
              cov = torch.exp(log_cov)
              predicted_distribution = MultimodalGaussianDistribution_torch(mu, cov, pi, args.n_modes_estimated, args.band_limit)
              energy = predicted_distribution.energy()
              predicted_density = predicted_distribution.density()
              mu_ = mu.detach()
              cov_ = cov.detach()
              error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
            
              measurement_cov = torch.ones_like(cov) * (args.measurement_noise **2)
              rmse_mu = root_mean_square_error_s1(mu_,ground_truth)
              rmse_mu_tot += rmse_mu
              rmse_cov += root_mean_square_error_s1(cov_,measurement_cov)
              mae_mu += mean_absolute_error(mu_,ground_truth)
            else:
              energy = model(ground_truth)
              predicted_density = torch.exp(energy - hed.normalization_constant(energy))
              mode = hed.mode(predicted_density,ground_truth)
              error, hist, bin_centers, bin_edges = predicted_residual_error(mode, ground_truth,20)
            
              rmse_mu = root_mean_square_error_s1(mode, ground_truth)
              rmse_mu_tot += rmse_mu
              mae_mu += mean_absolute_error(mode, ground_truth)

            if args.gaussianNLL == 0:
              if args.range_theta == None:
                loss = hed.negative_log_likelihood(energy, measurements)
              else:
                loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
            elif (args.gaussian_parameterisation and args.gaussianNLL) == 1:
              loss = predicted_distribution.negative_loglikelihood(measurements)
            else:
              raise ValueError("Invalid combination of parameters")
            mm_mean = positive_angle(ground_truth.unsqueeze(1) + centers) #(batch_size, n_modes, 2)
            true_distribution = MultimodalGaussianDistribution(mm_mean,measurement_noise,args.weights, args.n_modes,args.band_limit)
            true_distribution_energy = true_distribution.energy()
            if args.range_theta == None:
              true_distribution_density = true_distribution.density()
            else:
              true_distribution_density = true_distribution.density_local(args.range_theta)
            
            kl_div_tot += kl_divergence_s1(true_distribution_density, predicted_density)
            cdf_predicted_density = compute_cdf_from_pdf(predicted_density)
            cdf_true_density = compute_cdf_from_pdf(true_distribution_density)
            ece_tot += expected_calibration_error_continuous(cdf_true_density,cdf_predicted_density,20)
            wd_tot += wasserstein_distance(predicted_density, true_distribution_density)
            
            if epoch % 10 == 0 and i % log_freq == 0:
              indices = np.random.choice(args.batch_size, 5, replace=False)
              for j in indices:
                fig, ax = plt.subplots()
                if args.range_theta == None:
                  ax = plot_circular_distribution(energy[j],legend="predicted distribution",ax=ax)
                else:
                  ax = plot_circular_distribution(energy[j],legend="predicted distribution",ax=ax,mean=ground_truth[j,0],range_theta=args.range_theta)
                ax = plot_circular_distribution(true_distribution_energy[j],legend="true distribution",ax=ax)
                ax.plot(torch.cos(measurements[j]),torch.sin(measurements[j]),'o',label="measurement data")
                ax.plot(torch.cos(ground_truth[j]), torch.sin(ground_truth[j]), 'o', label="pose data")
                ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
                ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
                ax.set_aspect('equal')
                plt.savefig(os.path.join(logging_path, f"training_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
                # plt.show()
                plt.close()
              plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
              plt.axvline(x=rmse_mu, color='r', linestyle='-', label='RMSE')
              plt.legend()
              plt.xlabel('Value')
              plt.ylabel('Count')
              plt.title('Histogram predicted residual error')
              plt.savefig(os.path.join(logging_path, f"training_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
              plt.close()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tot += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {loss_tot / len(train_loader)}, KL divergence: {kl_div_tot / len(train_loader)}")
        metrics = {
          'Epoch': epoch + 1,
          'NLL Loss': loss_tot / len(train_loader),
          'KL Divergence': kl_div_tot / len(train_loader),
          'RMSE Mu': rmse_mu_tot / len(train_loader),
          'MAE Mu':mae_mu / len(train_loader),
          'ECE': ece_tot / len(train_loader), 
          'Wasserstein Distance': wd_tot / len(train_loader)}
        if args.gaussian_parameterisation:
          metrics['RMSE Cov'] = rmse_cov / len(train_loader)
        wandb.log(metrics)
    for img_file in os.listdir(logging_path):
        if img_file.endswith(".png"):
            wandb.log({f"{img_file}": wandb.Image(os.path.join(logging_path, img_file))})
    print("Training finished!")


if __name__ == '__main__':
  args = parse_args()
  if args.range_theta == "None" or args.range_theta == None:
      range_theta = None  # Return as is if the string is "None"
  else:
      range_theta=float(args.range_theta)  # Attempt to convert to float
  args.range_theta = range_theta
  torch.manual_seed(args.seed)
  # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)
  wandb.init(project="Diff-HEF",group="S1",entity="korra141",
            tags=["S1","UncertainityEstimation","MultimodalNoise"],
            name="S1-UncertainityEstimation",
            config=args)
  main(args)
