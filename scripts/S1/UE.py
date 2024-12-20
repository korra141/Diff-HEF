"""
This file contains an experiment to test whether its possible to learn the measueremnt distribution in S1 using
the Harmonic Exponential Functions. There is no process model and the measurement distribution is learned using 
the true position as input. 

Caveats:
- Since there are multiple trajectories that overlap each other, there is additional ambiguity about which distribution
the measurements are coming from. This is expected to increase the variance of the learned distribution or in the best case
scenario lead to multiple modes in the learned distribution.
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
import random
import pdb
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.utils.visualisation import plot_circular_distribution, plotting_von_mises
from src.data_generation.S1.toy_dataset import generating_data_S1_unimodal
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution,VonMissesDistribution_torch
from src.utils.metrics import kl_divergence_s1,root_mean_square_error_s1,mean_absolute_error, expected_calibration_error_continuous, compute_cdf_from_pdf, predicted_residual_error, wasserstein_distance,expected_calibration_error, sharpness_discrete


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment to learn the measurement distribution in S1 using Harmonic Exponential Functions.')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=10, help='Number of neurons in the hidden layer')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--band_limit', type=int, default=30, help='Band limit')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--measurement_noise', type=float, default=0.2, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument('--range_theta', type=str, default="None", help='Range of theta for local grid')
    parser.add_argument('--gaussian_parameterisation', type=int, default=0, help='Use Gaussian parameterisation')
    parser.add_argument('--gaussianNLL', type=int, default=0, help='Use Gaussian Negative Log Likelihood')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', type=int, default=1, help='Use wandb for logging')
    parser.add_argument('--learning_rate_start', type=float, default=0.001, help='Initial learning rate for decay')
    parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Final learning rate for decay')
    parser.add_argument('--lambda_start', type=float, default=100, help='Initial lambda for regularization')
    parser.add_argument('--lr_factor', type=float, default=10, help='Factor for learning rate decay')
    parser.add_argument('--reg_factor', type=float, default=0, help='Factor for regularization decay')
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
        # epsilon = torch.tensor(-1e-2)  # Small value to avoid zero covariance
        # logcov = torch.where(logcov < epsilon, epsilon, logcov)
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
  val_mae_mu_tot = 0
  ece_tot = 0
  sharpness = 0
  wd_tot = 0
  true_nll = 0
  with torch.no_grad():
    for i, (ground_truth, measurements) in enumerate(val_loader):
      if args.gaussian_parameterisation:
        mu, log_cov = model(ground_truth)
        cov = torch.exp(log_cov)
        predicted_distribution = VonMissesDistribution_torch(mu, cov, args.band_limit)
        predicted_density = predicted_distribution.density()
        energy = predicted_distribution.energy()
        mu_ = mu.detach()
        cov_ = cov.detach()
        error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
        measurement_cov = torch.ones_like(cov) * (args.measurement_noise **2)
        val_rmse_mu = root_mean_square_error_s1(mu_, ground_truth)
        val_rmse_mu_tot += val_rmse_mu
        val_rmse_cov += root_mean_square_error_s1(cov_, measurement_cov)
        val_mae_mu = mean_absolute_error(mu_, ground_truth)
        val_mae_mu_tot += val_mae_mu
      else:
        energy = model(ground_truth)
        predicted_density = torch.exp(energy - hed.normalization_constant(energy))
        mode = hed.mode(predicted_density,ground_truth)
        error, hist, bin_centers, bin_edges = predicted_residual_error(mode, ground_truth,20)
        val_rmse_mu = root_mean_square_error_s1(mode, ground_truth)
        val_rmse_mu_tot += val_rmse_mu
        val_mae_mu = mean_absolute_error(mode, ground_truth)
        val_mae_mu_tot += val_mae_mu
 
      if args.gaussianNLL == 0:
        if args.range_theta is None:
          loss = hed.negative_log_likelihood(energy, measurements)
        else:
          loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
        
      elif args.gaussian_parameterisation and args.gaussianNLL:
        loss = predicted_distribution.negative_loglikelihood(measurements)
      else:
        raise ValueError("Invalid combination of parameters")
      
      true_distribution = VonMissesDistribution(ground_truth.numpy(), args.measurement_noise, args.band_limit)
      if args.range_theta is None:
        true_density = true_distribution.density()
      else:
        true_density = true_distribution.density_local(args.range_theta)
      true_nll += true_distribution.negative_loglikelihood(measurements.numpy())
      val_kl_div_tot += kl_divergence_s1(true_density, predicted_density)
      ece = expected_calibration_error(predicted_density, true_density, M=10)
      ece_tot += ece
      sharpness += sharpness_discrete(predicted_density)
      wd_tot += wasserstein_distance(predicted_density, true_density)
      val_loss_tot += loss.item()
      if args.wandb and epoch % 50 == 0 and i == 0:
        indices = np.random.choice(args.batch_size, 5, replace=False)
        for j in indices:
          fig, ax = plt.subplots()
          if args.range_theta == None and args.gaussian_parameterisation:
            ax = plotting_von_mises(mu_[j],cov_.numpy()[j], args.band_limit,ax,"predicted distribution")
          elif args.range_theta == None and args.gaussian_parameterisation == 0:
            ax = plot_circular_distribution(energy[j],legend="predicted distribution",ax=ax)
          else:
            ax = plot_circular_distribution(energy[j],legend="predicted distribution",mean=ground_truth[j,0],range_theta=args.range_theta,ax=ax)
          ax = plotting_von_mises(ground_truth[j],args.measurement_noise **2, args.band_limit,ax,"true distribution")
          ax.plot(torch.cos(measurements[j]),torch.sin(measurements[j]),'o',label="measurement data")
          ax.plot(torch.cos(ground_truth[j]), torch.sin(ground_truth[j]), 'o', label="pose data")
          ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
          ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
          ax.set_aspect('equal')
          plt.savefig(os.path.join(logging_path, f"validation_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
          # plt.show()
          plt.close()
        fig, ax = plt.subplots()
          # Plot the histogram
        plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
        plt.axvline(x=val_mae_mu, color='r', linestyle='-', label='MAE')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram predicted residual error')
        plt.savefig(os.path.join(logging_path, f"validation_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
        # plt.show()
        plt.close()

    val_metrics = {
      'Epoch': epoch,
      "True NLL": true_nll / len(val_loader),
      'Validation NLL': val_loss_tot / len(val_loader),
      'Validation KL Divergence': val_kl_div_tot / len(val_loader),
      'Validation RMSE Mu': val_rmse_mu_tot / len(val_loader),
      'Validation MAE Mu' : val_mae_mu_tot / len(val_loader),
      'Validation ECE': ece_tot / len(val_loader),
      'Validation Sharpness': sharpness / len(val_loader),
      'Validation Wasserstein Distance': wd_tot / len(val_loader)}
    if args.gaussian_parameterisation:
        val_metrics['Validation RMSE Cov'] = val_rmse_cov / len(val_loader)
    
    return val_metrics


def main(args):

    if args.range_theta is not None and (args.gaussian_parameterisation == 1 or args.gaussianNLL == 1):
      raise ValueError("Invalid combination of parameters: range_theta cannot be used with gaussian_parameterisation or gaussianNLL")
    # Generate training data
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_unimodal(data_path,args.batch_size, args.n_samples, args.trajectory_length, args.measurement_noise, args.step_size, True)
    val_loader =  generating_data_S1_unimodal(data_path,args.batch_size, args.batch_size, args.trajectory_length, args.measurement_noise, args.step_size, True)
    
    # Initialize the model, optimizer, and loss function
    if args.gaussian_parameterisation:
        model = SimpleModel()
    else:
      model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    
    lambda_scheduler = lambda epoch: args.lambda_start * (torch.exp(torch.tensor(-args.reg_factor * epoch / args.num_epochs)) - 1)
    if args.reg_factor == 0:
       lambda_ = 0
    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    random_number = random.randint(1000, 9999)

    if args.range_theta == None and args.gaussian_parameterisation == 0:
        run_name = "S1-UE"    
    elif args.range_theta == None and args.gaussian_parameterisation == 1:
        run_name = "S1-UE-Gaussian"
    else:
        run_name = "S1-UE-Local"

    logging_path = os.path.join(base_path,"logs", run_name, str(ctime) + "_" + str(random_number))
    print(f"Logging path: {logging_path}")
    os.makedirs(logging_path)
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)
 
    # Training loop
    print("Training started!")
    start_time = time.time()
    for epoch in range(args.num_epochs):
      loss_tot = 0
      kl_div_tot = 0
      rmse_mu_tot = 0
      rmse_cov = 0
      mae_mu_tot = 0
      ece_tot = 0
      wd_tot = 0
      true_nll = 0
      nll_tot = 0
      sharpness = 0
      if args.lr_factor:
        learning_rate_decay = args.learning_rate_start + (args.learning_rate_end - args.learning_rate_start) * (1 - np.exp((-args.lr_factor * epoch / args.num_epochs)))
        print("learning_rate",learning_rate_decay)
        for param_group in optimizer.param_groups:
          param_group['lr'] = learning_rate_decay
      if args.reg_factor:
        lambda_ = lambda_scheduler(epoch)
        print("loss regulariser",lambda_)
      start_epoch_time = time.time()
      # val_metrics = validate(model, val_loader, args, hed,logging_path,epoch)
      # if args.wandb:
      #   wandb.log(val_metrics,commit=False)
      sample_batch = np.random.choice(len(train_loader),1).item()
      for i, (ground_truth, measurements) in enumerate(train_loader):
          if args.gaussian_parameterisation:
            mu, log_cov = model(ground_truth)
            cov = torch.exp(log_cov)
            predicted_distribution = VonMissesDistribution_torch(mu, cov,args.band_limit)
            energy = predicted_distribution.energy()
            predicted_density = predicted_distribution.density()
            mu_ = mu.detach()
            cov_ = cov.detach()
            error, hist, bin_centers, bin_edges = predicted_residual_error(mu_, ground_truth,20)
            measurement_cov = torch.ones_like(cov) * (args.measurement_noise **2)
            rmse_mu = root_mean_square_error_s1(mu_,ground_truth)
            rmse_mu_tot += rmse_mu
            rmse_cov += root_mean_square_error_s1(cov_,measurement_cov)
            mae_mu = mean_absolute_error(mu_,ground_truth)
          else:
            energy = model(ground_truth)
            predicted_density = torch.exp(energy - hed.normalization_constant(energy))
            mode = hed.mode(predicted_density,ground_truth)
            error, hist, bin_centers, bin_edges = predicted_residual_error(mode, ground_truth,20)
            rmse_mu = root_mean_square_error_s1(mode, ground_truth)
            rmse_mu_tot += rmse_mu
            mae_mu = mean_absolute_error(mode, ground_truth)
          mae_mu_tot += mae_mu

        # Compute the loss
          if args.gaussianNLL == 0:
            if args.range_theta == None:
              nll = hed.negative_log_likelihood(energy, measurements)
              loss = hed.loss_regularisation_norm(lambda_,energy,measurements,scaling_factor=1/args.lambda_start)
            else:
              nll = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
              loss = hed.loss_regularisation_norm(lambda_,energy,measurements,ground_truth,scaling_factor=1/args.lambda_start)

          elif (args.gaussian_parameterisation and args.gaussianNLL) == 1:
            loss = predicted_distribution.negative_loglikelihood(measurements)
            nll = loss
          else:
            raise ValueError("Invalid combination of parameters")
          true_distribution = VonMissesDistribution(ground_truth.numpy(), args.measurement_noise,args.band_limit)
          if args.range_theta == None:
            true_density = true_distribution.density()
          else:
            true_density = true_distribution.density_local(args.range_theta)
          ece = expected_calibration_error(predicted_density, true_density, M=10)
          ece_tot += ece
          sharpness += sharpness_discrete(predicted_density)
          true_nll += true_distribution.negative_loglikelihood(measurements.numpy())
          kl_div_tot += kl_divergence_s1(true_density, predicted_density)
          wd_tot += wasserstein_distance(predicted_density, true_density)
          # if args.wandb and epoch % 50 == 0 and i == sample_batch:
          #   indices = np.random.choice(args.batch_size, 5, replace=False)
          #   for j in indices:
          #     fig, ax = plt.subplots()
          #     if args.range_theta == None and args.gaussian_parameterisation:
          #       ax = plotting_von_mises(mu_[j],cov_.numpy()[j], args.band_limit,ax,"predicted distribution")
          #     elif args.range_theta == None and args.gaussian_parameterisation == 0:
          #       ax = plot_circular_distribution(energy[j],legend="predicted distribution",ax=ax)
          #     else:
          #       ax = plot_circular_distribution(energy[j],legend="predicted distribution",mean=ground_truth[j,0],range_theta=args.range_theta,ax=ax)
          #     ax = plotting_von_mises(ground_truth[j],args.measurement_noise**2 , args.band_limit,ax,"true distribution")
          #     ax.plot(torch.cos(measurements[j]),torch.sin(measurements[j]),'o',label="measurement data")
          #     ax.plot(torch.cos(ground_truth[j]), torch.sin(ground_truth[j]), 'o', label="pose data")
          #     ax.set_title(f"Epoch {epoch} Batch {i} Sample {j}", loc='center')
          #     ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
          #     ax.set_aspect('equal')
          #     plt.savefig(os.path.join(logging_path, f"training_epoch_{epoch}_batch_{i}_sample_{j}.png"), format='png', dpi=300)
          #     # plt.show()
          #     plt.close()
          #   plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
          #   plt.axvline(x=mae_mu, color='r', linestyle='-', label='MAE')
          #   plt.legend()
          #   plt.xlabel('Value')
          #   plt.ylabel('Count')
          #   plt.title('Histogram predicted residual error')
          #   plt.savefig(os.path.join(logging_path, f"training_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
          #   plt.close()


        # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_tot += loss.item()
          nll_tot += nll.item()
      print(f"Epoch {epoch+1} took {time.time() - start_epoch_time} seconds")
      print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss_tot/len(train_loader):.4f}, RMSE Mu: {rmse_mu_tot/len(train_loader):.4f}, MAE Mu: {mae_mu_tot/len(train_loader):.4f}, KL Divergence: {kl_div_tot/len(train_loader):.4f}')
      print(f"ECE: {ece_tot/len(train_loader):.4f}, Sharpness: {sharpness/len(train_loader):.4f}")
      
      if args.wandb and epoch % 50 == 0:
        metrics = {
          'Epoch': epoch + 1,
          'NLL' : nll_tot / len(train_loader),
          'True NLL': true_nll / len(train_loader),
          'Loss': loss_tot / len(train_loader),
          'KL Divergence': kl_div_tot / len(train_loader),
          'RMSE Mu': rmse_mu_tot / len(train_loader),
          'MAE Mu':mae_mu_tot / len(train_loader),
          'ECE': ece_tot / len(train_loader), 
          'Sharpness': sharpness / len(train_loader),
          'Wasserstein Distance': wd_tot / len(train_loader)}
        if args.gaussian_parameterisation:
          metrics['RMSE Cov'] = rmse_cov / len(train_loader)
        wandb.log(metrics)
    print("Training finished!")
    # Save the model checkpoint
    model_save_path = os.path.join(logging_path, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    wandb.save(model_save_path)
    
    print(f"Training took {time.time() - start_time} seconds")
    # Log all the generated images to wandb
    # if args.wandb:
    #   print("Logging images to wandb")
    #   start_wandb_time = time.time()
    #   for img_file in os.listdir(logging_path):
    #       if img_file.endswith(".png"):
    #           wandb.log({f"{img_file}": wandb.Image(os.path.join(logging_path, img_file))})
    #   print(f"Logging images to wandb took {time.time() - start_wandb_time} seconds")
    


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
  if args.wandb:
    run = wandb.init(project="Diff-HEF",group="S1",entity="korra141",
              tags=["S1","UncertainityEstimation","UnimodalNoise"],
              name="S1-UncertainityEstimation",
              config=args)
    artifact = wandb.Artifact("UE_script_1", type="script")
    artifact.add_file(__file__)
    run.log_artifact(artifact)
  main(args)