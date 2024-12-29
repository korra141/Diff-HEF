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
import pickle as pkl 

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.data_generation.S1.toy_dataset import generating_data_S1_multimodal
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.S1.WrappedNormalDitribution import MultimodalGaussianDistribution,MultimodalGaussianDistribution_torch,MultimodalWrappedNormalDistribution
from src.utils.visualisation import plot_circular_distribution
from src.utils.metrics import kl_divergence_s1,root_mean_square_error_s1,mean_absolute_error, expected_calibration_error_continuous, compute_cdf_from_pdf, predicted_residual_error, wasserstein_distance,expected_calibration_error, sharpness_discrete

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment to learn the measurement distribution in S1 using Harmonic Exponential Functions.')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=10, help='Number of neurons in the hidden layer')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--band_limit', type=int, default=60, help='Band limit')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--measurement_noise', type=float, default=0.2, help='Standard deviation of the measurement noise')
    # parser.add_argument('--measurement_noise_max', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument("--mean_offset", type=float, default=0.785, help="Mean offset for the multimodal noise")
    parser.add_argument("--n_modes", type=int, default=2, help="Number of modes for the multimodal noise")
    parser.add_argument("--n_modes_estimated", type=int, default=10, help="Number of modes to be estimated by the model")
    parser.add_argument('--range_theta', type=str, default="None", help='Range of theta for local grid')
    parser.add_argument('--gaussian_parameterisation', type=int, default=0, help='Use Gaussian parameterisation')
    parser.add_argument('--gaussianNLL', type=int, default=0, help='Use Gaussian Negative Log Likelihood')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate_start', type=float, default=0.001, help='Initial learning rate for decay')
    parser.add_argument('--learning_rate_end', type=float, default=0.00001, help='Final learning rate for decay')
    parser.add_argument('--lambda_start', type=float, default=100, help='Initial lambda for regularization')
    parser.add_argument('--lr_factor', type=float, default=10, help='Factor for learning rate decay')
    parser.add_argument('--reg_factor', type=float, default=0, help='Factor for regularization decay')
    
    # parser.add_argument('--weights', type=str, default=[0.5,0.5], help='Path to the model weights')
    
    return parser.parse_args()  
def positive_angle(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

# Define the neural network architecture
class EnergyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 20)
        self.fc3 = nn.Linear(20, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out
        #     self.model = nn.Sequential(
    #         nn.Linear(1, 8),
    #         nn.ReLU(),
    #         nn.Linear(8, 3*n_modes)
    #     )
    #     self._initialize_weights()

    # def forward(self, x):
    #     y_pred = self.model(x)
    #     mu = y_pred[:,0:self.n_modes] %(2 * math.pi)
    #     logcov = y_pred[:,self.n_modes:2*self.n_modes]
    #     pi = F.softmax(y_pred[:,2*self.n_modes:],dim=1)
        
    #     epsilon = torch.tensor(-1e-2)  # Small value to avoid zero covariance
    #     logcov = torch.where(logcov < epsilon, epsilon, logcov)
    #     return mu, logcov, pi
class SimpleModel(nn.Module):
    def __init__(self,input_dim, hidden_dim,n_modes):
        super(SimpleModel, self).__init__()
        self.n_modes = n_modes
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.pi = nn.Linear(hidden_dim, self.n_modes)  # Mixing coefficients
        self.mu = nn.Linear(hidden_dim, self.n_modes)  # Means
        self.cov = nn.Linear(hidden_dim, self.n_modes)  # Standard deviations
        # self._initialize_weights()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.hidden(x)
        # pi_semi = self.sigmoid(self.pi(h))
        # pi = torch.nn.functional.softmax(pi_semi, dim=-1)  # Mixing coefficients (softmax for probabilities)
        # mu = self.mu(h)  # Means
        # sigma = torch.clip(self.sigma(h)**2 + 1e-3, 0, 100) # Standard deviatcons (exp for positivity)
        # mu = torch.remainder(mu, 2*torch.pi)
        # sigma = torch.remainder(sigma, 2*torch.pi)
        # return mu, sigma, pi, h, pi_semi
    
        pi = torch.nn.functional.softmax(h, dim=-1)  # Mixing coefficients (softmax for probabilities)
        mu = self.sigmoid(self.mu(h)) * (2*math.pi)  # Means
        # sigma = torch.exp(self.sigma(h))
        # sigma = torch.nn.functional.softplus(self.sigma(h)) + 1e-6
        cov = torch.clip(self.cov(h) + 1e-3, -10, 10) # Standard deviatcons
        return mu, cov, pi
    
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
  val_mae_mu_tot = 0
  ece_tot = 0
  sharpness = 0
  wd_tot = 0
  true_nll = 0
  with torch.no_grad():
    for i, (ground_truth, measurements) in enumerate(val_loader):
      mm_mean = positive_angle(ground_truth.unsqueeze(1) + centers)
      if args.gaussian_parameterisation:
        mu, logcov, pi = model(ground_truth)
        cov = torch.nn.functional.softplus(logcov) + 1e-6
        predicted_distribution = MultimodalGaussianDistribution_torch(mu, cov, pi, args.n_modes_estimated, args.band_limit)
        energy = predicted_distribution.energy()
        predicted_density = predicted_distribution.density()
        mu_ = mu.detach().unsqueeze(-1)
        # if mu_.shape[-1] != args.n_modes:
        #   mu_temp,_ = torch.topk(predicted_density, args.n_modes, dim=1)
        #   mu_temp = mu_temp.detach().unsqueeze(-1)
        # else:
        #   mu_temp = mu_
        # sigma_ = sigma.detach()
        # error, hist, bin_centers, bin_edges = predicted_residual_error(mu_temp, mm_mean,20)
        # measurement_sigma = torch.ones_like(sigma) * (args.measurement_noise)
        # val_rmse_mu = root_mean_square_error_s1( mu_temp,mm_mean)
        # val_rmse_mu_tot += val_rmse_mu
        # val_rmse_cov += root_mean_square_error_s1(sigma_, measurement_sigma)
        # val_mae_mu = mean_absolute_error( mu_temp,mm_mean)
      else:
        energy = model(ground_truth)
        predicted_density = torch.exp(energy - hed.normalization_constant(energy))
        mode = hed.mode(predicted_density,ground_truth,n_modes=args.n_modes)
      #   error, hist, bin_centers, bin_edges = predicted_residual_error(mode, mm_mean,20)
      #   val_rmse_mu = root_mean_square_error_s1(mode, mm_mean)
      #   val_rmse_mu_tot += val_rmse_mu
      #   val_mae_mu = mean_absolute_error(mode, mm_mean)
      # val_mae_mu_tot += val_mae_mu
      
      if args.gaussianNLL == 0:
        if args.range_theta is None:
          loss = hed.negative_log_likelihood(energy, measurements)
        else:
          loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
        
      elif args.gaussian_parameterisation and args.gaussianNLL:
        loss = predicted_distribution.negative_loglikelihood(measurements)
      else:
        raise ValueError("Invalid combination of parameters")
      
      true_distribution = MultimodalGaussianDistribution(mm_mean, measurement_noise, args.weights, args.n_modes, args.band_limit)
      true_distribution_energy = true_distribution.energy()
      if args.range_theta is None:
        true_density = true_distribution.density()
      else:
        true_density = true_distribution.density_local(args.range_theta)
      true_nll += true_distribution.negative_loglikelihood(measurements)
      val_kl_div_tot += kl_divergence_s1(true_density, predicted_density)
      wd_tot += wasserstein_distance(predicted_density, true_density)
      
      val_loss_tot += loss.item()
      if (epoch % 100 == 0 or epoch == args.num_epochs -1) and i == 0:
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
          plot_dict = {}
          plot_dict["groundtruth"] = ground_truth[j]
          plot_dict["measurement"] = measurements[j]
          plot_dict["true_dist_energy"] = true_distribution_energy[j]
          plot_dict["predicted_dist_energy"] = energy[j]
          with open(os.path.join(logging_path, f"validation_epoch_{epoch}_batch_{i}_sample_{j}.pkl"), "wb") as f:
                    pkl.dump(plot_dict, f)
                    print(os.path.join(logging_path, f"validation_epoch_{epoch}_batch_{i}_sample_{j}.pkl"))

        '''
        plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
        plt.axvline(x=val_mae_mu, color='r', linestyle='-', label='MAE')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram predicted residual error')
        plt.savefig(os.path.join(logging_path, f"validation_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
        # plt.show()
        plt.close()
        '''
    val_metrics = {
      'Epoch': epoch,
      "True NLL": true_nll / len(val_loader),
      'Validation NLL': val_loss_tot / len(val_loader),     
      'Validation KL Divergence': val_kl_div_tot / len(val_loader),
      #'Validation RMSE Mu': val_rmse_mu_tot / len(val_loader),
      #'Validation MAE Mu' : val_mae_mu / len(val_loader),
      'Validation ECE': ece_tot / len(val_loader),
      'Validation Sharpness': sharpness / len(val_loader),
      'Validation Wasserstein Distance': wd_tot / len(val_loader)}
    #if args.gaussian_parameterisation:
    #    val_metrics['Validation RMSE Cov'] = val_rmse_cov / len(val_loader)
    
    wandb.log(val_metrics,commit=False) 


def main(args):
    
    if args.range_theta is not None and (args.gaussian_parameterisation == 1 or args.gaussianNLL == 1):
      raise ValueError("Invalid combination of parameters: range_theta cannot be used with gaussian_parameterisation or gaussianNLL")
   
    measurement_noise = torch.ones(args.n_modes) * args.measurement_noise
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_multimodal(measurement_noise, args.mean_offset,args.n_modes, data_path, args.batch_size, args.n_samples, args.trajectory_length,  args.step_size, True)
    val_loader =  generating_data_S1_multimodal(measurement_noise, args.mean_offset,args.n_modes, data_path, args.batch_size, args.batch_size, args.trajectory_length,  args.step_size, True)

    if args.gaussian_parameterisation:
      model = SimpleModel(1, args.hidden_size,args.n_modes_estimated)
    else:
      model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_start)
    centers = torch.tile(torch.linspace(-args.mean_offset / 2, args.mean_offset / 2, args.n_modes)[None,:,None], (args.batch_size, 1,1)) # n_modes
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
    lambda_scheduler = lambda epoch: args.lambda_start * (torch.exp(torch.tensor(-args.reg_factor * epoch / args.num_epochs)) - 1)
    if args.reg_factor == 0:
       lambda_ = 0
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
    args.weights = torch.ones(args.n_modes) / args.n_modes
    # Loss function
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)
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
      # validate(model, val_loader, args, hed, centers,measurement_noise,logging_path,epoch)
      sample_batch = np.random.choice(len(train_loader),1).item()
      for i, (ground_truth, measurements) in enumerate(train_loader):
            start_time = time.time()
            mm_mean = positive_angle(ground_truth.unsqueeze(1) + centers)
            # Forward pass
            if args.gaussian_parameterisation:
              mu, logcov, pi = model(ground_truth)
              cov = torch.nn.functional.softplus(logcov) + 1e-6
              predicted_distribution = MultimodalWrappedNormalDistribution(mu, cov, pi, args.n_modes_estimated, args.band_limit)
              if(torch.isnan(pi).any()):
                print("pi is inf")
              # cov  = torch.exp(log_cov)
              # predicted_distribution = MultimodalGaussianDistribution_torch(mu, sigma, pi, args.n_modes_estimated, args.band_limit)
              energy = predicted_distribution.energy()
              predicted_density = predicted_distribution.density()
              mu_ = mu.detach().unsqueeze(-1)
              '''
              if mu_.shape[1] != args.n_modes:
                # mu_temp,_ = torch.topk(predicted_density, args.n_modes, dim=1)
                # mu_temp = mu_temp.detach().unsqueeze(-1)
                print(pi.size(), args.n_modes)
                pi_temp,indices = torch.topk(pi, args.n_modes, dim=1)
                batch_indices = torch.arange(mu_.size(0)).unsqueeze(1) 
                mu_temp = mu_[batch_indices,indices]
              else:
                mu_temp = mu_
              if torch.isnan(mu_temp).any():
                print("mu is inf")
                pdb.set_trace()
            
              sigma_ = sigma.detach()
              error, hist, bin_centers, bin_edges = predicted_residual_error(mu_temp, mm_mean,20)
            
              measurement_noise_true = torch.ones_like(sigma) * (args.measurement_noise)
              rmse_mu = root_mean_square_error_s1( mu_temp,mm_mean)
              rmse_mu_tot += rmse_mu
              rmse_cov += root_mean_square_error_s1(sigma_,measurement_noise_true)
              mae_mu = mean_absolute_error(mu_temp,mm_mean)
              '''
            else:
              energy = model(ground_truth)
              predicted_density = torch.exp(energy - hed.normalization_constant(energy))
              mode = hed.mode(predicted_density,ground_truth,n_modes=args.n_modes)
              # print(mode.shape,mm_mean.shape)
              '''
              error, hist, bin_centers, bin_edges = predicted_residual_error(mode, mm_mean,20)
            
              rmse_mu = root_mean_square_error_s1(mode, mm_mean)
              rmse_mu_tot += rmse_mu
              mae_mu = mean_absolute_error(mode, mm_mean)
            mae_mu_tot += mae_mu
            '''
            if args.gaussianNLL == 0:
              if args.range_theta == None:
                nll = hed.negative_log_likelihood(energy, measurements)
                loss = hed.loss_regularisation_norm(lambda_,energy,measurements,scaling_factor=1/args.lambda_start)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                  print("loss is nan")
                  # pdb.set_trace()
              else:
                nll = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
                loss = hed.loss_regularisation_norm(lambda_,energy,measurements,ground_truth,scaling_factor=1/args.lambda_start)

            elif (args.gaussian_parameterisation and args.gaussianNLL) == 1:
              loss = predicted_distribution.negative_loglikelihood(measurements)
              nll = loss
            else:
              raise ValueError("Invalid combination of parameters")
             #(batch_size, n_modes, 1)
            true_distribution = MultimodalWrappedNormalDistribution(mm_mean,measurement_noise,args.weights, args.n_modes,args.band_limit)
            true_distribution_energy = true_distribution.energy_1cov()
            if args.range_theta == None:
              true_distribution_density = true_distribution.density_1cov()
            else:
              true_distribution_density = true_distribution.density_local(args.range_theta)
            
            ece = expected_calibration_error(predicted_density, true_distribution_density, M=10)
            ece_tot += ece
            sharpness += sharpness_discrete(predicted_density)
            true_nll += true_distribution.negative_loglikelihood_1cov(measurements)
            kl_div_tot += kl_divergence_s1(true_distribution_density, predicted_density)
            wd_tot += wasserstein_distance(predicted_density, true_distribution_density)
            
            if (epoch == args.num_epochs -1) and i  == sample_batch:
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
                plot_dict = {}
                plot_dict["groundtruth"] = ground_truth[j]
                plot_dict["measurement"] = measurements[j]
                plot_dict["true_dist_energy"] = true_distribution_energy[j]
                plot_dict["predicted_dist_energy"] = energy[j]
                with open(os.path.join(logging_path, f"training_epoch_{epoch}_batch_{i}_sample_{j}.pkl"), "wb") as f:
                    pkl.dump(plot_dict, f)
                    print(os.path.join(logging_path, f"training_epoch_{epoch}_batch_{i}_sample_{j}.pkl"))
                '''
                #plt.bar(bin_centers.numpy(), hist.numpy(), width=(bin_edges[1] - bin_edges[0]).item(), edgecolor='black', align='center')
              #plt.axvline(x=mae_mu, color='r', linestyle='-', label='MAE')
              #plt.legend()
              plt.xlabel('Value')
              plt.ylabel('Count')
              plt.title('Histogram predicted residual error')
              plt.savefig(os.path.join(logging_path, f"training_histogram_epoch_{epoch}_batch_{i}.png"), format='png', dpi=300)
              plt.close()
              '''

            # Backward pass and optimization
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            # grad_norm = 0
            # for name, param in model.named_parameters():
            #   if param.grad is not None:
            #     grad_norm += param.grad.norm()
             
            # if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any() :
            #   print("grad norm is nan or inf")
            #   pdb.set_trace()
            # if i % 500 ==0:
            #     print(f"Epoch {epoch}, Batch {i}, Gradient norm: {grad_norm}")
            optimizer.step()
            loss_tot += loss.item()
            nll_tot += nll.item()
      print(f"Epoch {epoch + 1}, Loss: {loss_tot / len(train_loader)}, KL divergence: {kl_div_tot / len(train_loader)}")
      print(f"True Nll: {true_nll / len(train_loader)}")
      if epoch % 100 == 0:
        metrics = {
            'Epoch': epoch + 1,
            'NLL' : nll_tot / len(train_loader),
            'True NLL': true_nll / len(train_loader),
            'Loss': loss_tot / len(train_loader),
            'KL Divergence': kl_div_tot / len(train_loader),
            #'RMSE Mu': rmse_mu_tot / len(train_loader),
            #'MAE Mu':mae_mu_tot / len(train_loader),
            'ECE': ece_tot / len(train_loader), 
            'Sharpness': sharpness / len(train_loader),
            'Wasserstein Distance': wd_tot / len(train_loader)}
          # Log all the generated images to wandb
  #if args.gaussian_parameterisation:
        #    metrics['RMSE Cov'] = rmse_cov / len(train_loader)
        wandb.log(metrics)
    print("Training finished!")
    model_save_path = os.path.join(logging_path, 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    wandb.save(model_save_path)
    for img_file in os.listdir(logging_path):
      print(img_file)
      if img_file.endswith(".pkl"):
          wandb.save(os.path.join(logging_path, img_file))
      if img_file.endswith(".png"):
          wandb.log({f"{img_file}": wandb.Image(os.path.join(logging_path, img_file))})

    print(f"Training took {time.time() - start_time} seconds")


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
  run = wandb.init(project="Diff-HEF",group="S1",entity="korra141",
            tags=["S1","UncertainityEstimation","MultimodalNoise"],
            name="S1-UncertainityEstimation",
            config=args)
  artifact = wandb.Artifact("UE_mm_script_7", type="mm_script")
  artifact.add_file(__file__)
  run.log_artifact(artifact)
  main(args)
