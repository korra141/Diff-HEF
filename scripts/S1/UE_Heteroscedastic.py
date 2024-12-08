"""
This file contains an experiment to test whether its possible to learn the measueremnt distribution in S1 using
the Harmonic Exponential Functions. There is no process model and the measurement distribution is learned using 
the true position as input. 

Caveats:
- Since there are multiple trajectories that overlap each other, there is additional ambiguity about which distribution
the measurements are coming from. This is expected to increase the variance of the learned distribution or in the best case
scenario lead to multiple modes in the learned distribution.


Same as Experiment 1 but with Heteroscedastic noise.
"""


import time
import torch
import argparse
import torch.fft
import torch.nn as nn
import torch.optim as optim
from scipy.special import i0

import math
import os   

import matplotlib.pyplot as plt
import numpy as np
from time import strftime, localtime
import wandb
import sys
import pdb

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.data_generation.S1.toy_dataset import generating_data_S1_heteroscedastic,heteroscedastic_noise
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution,VonMissesDistribution_torch
from src.utils.visualisation import plot_circular_distribution, plotting_von_mises
from src.utils.metrics import kl_divergence_s1,root_mean_square_error_s1,mean_absolute_error, expected_calibration_error_continuous, compute_cdf_from_pdf, predicted_residual_error, wasserstein_distance
import random
import string


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
    parser.add_argument('--measurement_noise_min', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--measurement_noise_max', type=float, default=0.3, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument('--shuffle_flag', type=bool, default=True, help='Whether to shuffle the samples')
    parser.add_argument('--range_theta', type=str, default="None", help='Range of theta for local grid')
    parser.add_argument('--gaussian_parameterisation', type=int, default=0, help='Use Gaussian parameterisation')
    parser.add_argument('--gaussianNLL', type=int, default=0, help='Use Gaussian Negative Log Likelihood')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()  

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
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self._initialize_weights()

    def forward(self, x):
        y_pred = self.model(x)
        mu = y_pred[:,0:1] %(2 * math.pi)
        logcov = y_pred[:,1:2]
        epsilon = torch.tensor(-1e-2)  # Small value to avoid zero covariance
        logcov = torch.where(logcov < epsilon, epsilon, logcov)
        return mu, logcov

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def validate(model, val_loader, args, hed,logging_path,epoch):
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
        mu, log_cov = model(ground_truth)
        cov = torch.exp(log_cov)
        predicted_distribution = VonMissesDistribution_torch(mu, cov, args.band_limit)
        energy = predicted_distribution.energy()
        predicted_density = predicted_distribution.density()
        mu_ = mu.detach()
        cov_ = cov.detach()
        error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
       
        measurement_cov = heteroscedastic_noise(ground_truth, args.measurement_noise_min, args.measurement_noise_max) ** 2
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
      
      true_distribution = VonMissesDistribution(ground_truth.numpy(), heteroscedastic_noise(ground_truth, args.measurement_noise_min, args.measurement_noise_max).numpy(), args.band_limit)
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
          if args.range_theta == None and args.gaussian_parameterisation:
            ax = plotting_von_mises(mu_[j],cov_.numpy()[j], args.band_limit,ax,"predicted distribution")
          elif args.range_theta == None and args.gaussian_parameterisation == 0:
            ax = plot_circular_distribution(energy[0],legend="predicted distribution",ax=ax)
          else:
            ax = plot_circular_distribution(energy[0],legend="predicted distribution",mean=ground_truth[0,0],range_theta=args.range_theta,ax=ax)
          ax = plotting_von_mises(ground_truth[0],heteroscedastic_noise(ground_truth[0], args.measurement_noise_min, args.measurement_noise_max).item()**2, args.band_limit,ax,"true distribution")
          ax.plot(torch.cos(measurements[0]),torch.sin(measurements[0]),'o',label="measurement data")
          ax.plot(torch.cos(ground_truth[0]), torch.sin(ground_truth[0]), 'o', label="pose data")
          ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
          ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
          ax.set_aspect('equal')
          plt.savefig(os.path.join(logging_path, f"validation_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
          # plt.show()
          plt.close()
        fig, ax = plt.subplots()
          # Plot the histogram
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

    # Generate training data
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_heteroscedastic(data_path, args, args.batch_size, args.n_samples, args.trajectory_length, args.measurement_noise_min, args.step_size, True)
    val_loader = generating_data_S1_heteroscedastic(data_path,args, args.batch_size, args.batch_size, args.trajectory_length, args.measurement_noise_min, args.step_size, False)
    log_freq = len(train_loader)/3
    # Initialize the model, optimizer, and loss function
    if args.gaussian_parameterisation:
        model = SimpleModel()
    else:
      model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    random_number = random.randint(1000, 9999)
    if args.range_theta == None and args.gaussian_parameterisation == 0:
        run_name = "S1-UE-Heteroscedastic"    
    elif args.range_theta == None and args.gaussian_parameterisation == 1:
        run_name = "S1-UE-Heteroscedastic-Gaussian"
    else:
        run_name = "S1-UE-Heteroscedastic-Local"

    logging_path = os.path.join(base_path,"logs", run_name, str(ctime) + "_" + str(random_number))
    os.makedirs(logging_path)
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)
    for epoch in range(args.num_epochs):
        loss_tot = 0
        kl_div_tot = 0
        rmse_mu_tot = 0
        rmse_cov = 0
        mae_mu = 0
        ece_tot = 0
        wd_tot = 0
        validate(model, val_loader, args, hed,logging_path,epoch)
        for i, (ground_truth, measurements) in enumerate(train_loader):
          if args.gaussian_parameterisation:
            mu, log_cov = model(ground_truth)
            cov = torch.exp(log_cov)
            predicted_distribution = VonMissesDistribution_torch(mu, cov,args.band_limit)
            energy = predicted_distribution.energy()
            predicted_density = predicted_distribution.density()
            
            mu_ = mu.detach()
            cov_ = cov.detach()
            measurement_cov = heteroscedastic_noise(ground_truth, args.measurement_noise_min, args.measurement_noise_max) ** 2
            error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
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
          true_distribution = VonMissesDistribution(ground_truth.numpy(), heteroscedastic_noise(ground_truth, args.measurement_noise_min, args.measurement_noise_max).numpy(),args.band_limit)
          if args.range_theta == None:
            true_density = true_distribution.density()
          else:
            true_density = true_distribution.density_local(args.range_theta)
          kl_div_tot += kl_divergence_s1(true_density, predicted_density)
          # Log the results
          cdf_predicted_density = compute_cdf_from_pdf(predicted_density)
          cdf_true_density = compute_cdf_from_pdf(true_density)
          ece_tot += expected_calibration_error_continuous(cdf_true_density,cdf_predicted_density,20)
          wd_tot += wasserstein_distance(predicted_density, true_density)
          if epoch % 10 == 0 and i % log_freq == 0:
            indices = np.random.choice(args.batch_size, 5, replace=False)
            for j in indices:
              fig, ax = plt.subplots()
              if args.range_theta == None and args.gaussian_parameterisation:
                ax = plotting_von_mises(mu_[j],cov_.numpy()[j], args.band_limit,ax,"predicted distribution")
              elif args.range_theta == None and args.gaussian_parameterisation == 0:
                ax = plot_circular_distribution(energy[0],legend="predicted distribution",ax=ax)
              else:
                ax = plot_circular_distribution(energy[0],legend="predicted distribution",mean=ground_truth[0,0],range_theta=args.range_theta,ax=ax)
              ax = plotting_von_mises(ground_truth[0],heteroscedastic_noise(ground_truth[0], args.measurement_noise_min, args.measurement_noise_max).item()**2, args.band_limit,ax,"true distribution")
              ax.plot(torch.cos(measurements[0]),torch.sin(measurements[0]),'o',label="measurement data")
              ax.plot(torch.cos(ground_truth[0]), torch.sin(ground_truth[0]), 'o', label="pose data")
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
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss_tot/len(train_loader):.4f}, RMSE Mu: {rmse_mu/len(train_loader):.4f}, MAE Mu: {mae_mu/len(train_loader):.4f}, KL Divergence: {kl_div_tot/len(train_loader):.4f}')
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
    # Log all the generated images to wandb
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
          tags=["S1","UncertainityEstimation","HeteroscedasticNoise"],
          name="S1-UncertainityEstimation",
          config=args)
  main(args)