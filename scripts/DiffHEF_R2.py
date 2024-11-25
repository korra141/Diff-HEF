
"""
The script uses a CNN model to estimate the density of measurement given input pose.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
import pdb
import random
import torch
import torch.optim as optim
import os
import datetime
import argparse
import wandb
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

# local import 
from src.utils.visualisation import plot_3d_filter,generate_gif,plot_3d
from src.distributions.R2.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.R2.StandardDistribution import GaussianDistribution
from src.utils.debug_tools import check_model_weights_nan, check_tensor_nan, get_gradients
from src.data_generation.R2.toy_dataset import TrajectoryGenerator
from src.models.CNN import LikelihoodPredicitionCNN,init_weights_identity,init_weights_zero
from src.utils.metrics import kl_divergence, mean_absolute_error, root_mean_square_error
from src.filter.HEF import HEFilter

def validate_model(model, data_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise, hef,trajectory_length,step_t,motion_noise):
  model.eval()
  val_dict = {}
  with torch.no_grad():
    val_loss_tot = 0
    val_kl_2_tot = 0
    val_true_nll_tot = 0
    val_mae_tot = 0
    val_rmse_tot = 0
    val_measurement_nll_tot = 0
    for val_batch_idx, (val_ground_truth, val_measurements) in enumerate(data_loader):
      print(f"Validation Batch: {val_batch_idx}")
      prior = GaussianDistribution(val_ground_truth[:, 0], initial_noise, range_x, range_y, band_limit)
      prior_pdf = prior.density_over_grid()
      control = torch.ones((val_ground_truth.size(0), 2)) * torch.tensor(step_t) + np.random.normal(loc=0, scale=measurement_noise, size=(val_ground_truth.size(0), 2))
      for j in range(trajectory_length):
        input_measurement = GaussianDistribution(val_measurements[:, j], initial_noise, range_x, range_y, band_limit)
        input_measurement_pdf = input_measurement.density_over_grid()

        process = GaussianDistribution(control, motion_noise, range_x, range_y, band_limit)
        process_pdf = process.density_over_grid()

        # Predict Step
        eta_bel_x_t_bar, density_bel_x_t_bar = hef.predict(prior_pdf, process_pdf)
        # P(z_t | x_t) Measurement Step
        density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32)
        measurement_pdf = model(input_measurement_pdf, density_bel_x_t_bar)

        z = hed.normalization_constant(measurement_pdf)
        measurement_pdf = measurement_pdf / z

        # Calculate Posterior
        posterior_pdf = hef.update(eta_bel_x_t_bar, measurement_pdf)
        posterior_pdf = posterior_pdf.to(torch.float32)

        # Current posterior is prior for the next step (j + 1)
        prior_pdf = posterior_pdf

        val_loss = hed.negative_log_likelihood_density(posterior_pdf, val_ground_truth[:, j])
        target_distribution = GaussianDistribution(val_measurements[:,j], measurement_noise, range_x, range_y, band_limit)
        val_true_nll_input = target_distribution.negative_log_likelihood(val_measurements)
        val_true_density = target_distribution.density_over_grid()
        val_measurement_nll = hed.negative_log_likelihood_density(measurement_pdf, val_measurements[:, j])

        mode = hed.mode(posterior_pdf)
        val_rmse_tot += root_mean_square_error(mode, val_ground_truth[:,j]).item()
        val_mae_tot += mean_absolute_error(mode, val_ground_truth[:,j]).item()
        val_kl_div_2 = kl_divergence(val_true_density, measurement_pdf)

        val_loss_tot += val_loss.item()
        val_kl_2_tot += val_kl_div_2.item()
        val_true_nll_tot += val_true_nll_input.item()
        val_measurement_nll_tot += val_measurement_nll.item()
    val_dict = {
      'Val Posterior loss': val_loss_tot / (len(data_loader) * trajectory_length),
      'Val KL Div': val_kl_2_tot / (len(data_loader) * trajectory_length),
      'Val True NLL': val_true_nll_tot / (len(data_loader) * trajectory_length),
      'Val MAE': val_mae_tot / (len(data_loader) * trajectory_length),
      'Val RMSE': val_rmse_tot / (len(data_loader) * trajectory_length),
      'Val Measurement NLL': val_measurement_nll_tot / (len(data_loader) * trajectory_length)
    }
  return val_dict
  
def main(args):
  # Data Parameters
  n_samples = args.n_samples
  trajectory_length = args.trajectory_length
  measurement_noise = args.measurement_noise
  initial_noise = args.initial_noise
  range_x = (args.range_x_start, args.range_x_end)
  range_y = (args.range_y_start, args.range_y_end)
  band_limit = args.band_limit
  motion_noise = args.motion_noise

  range_x_diff = range_x[1] - range_x[0]
  range_y_diff = range_y[1] - range_y[0]
  step_t = (round(range_x_diff/band_limit[0], 2), round(range_y_diff/band_limit[1], 2))
  sample_idx = 0

  # Training Parameters
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  num_epochs = args.num_epochs

  data_path = os.path.join(base_path, 'data')
  data_generator = TrajectoryGenerator(range_x, range_y, step_t, n_samples , trajectory_length, measurement_noise)
  train_loader, val_loader = data_generator.create_data_loaders(data_path, batch_size,flag_flattend=False)
  
  measurement_model = LikelihoodPredicitionCNN(band_limit,batch_size) # Assuming input and output dimensions are 3
  measurement_model.apply(init_weights_identity)

  if args.optimizer == 'adam':
      optimizer = optim.Adam(measurement_model.parameters(), lr=learning_rate)
  elif args.optimizer == 'rmsprop':
      optimizer = optim.RMSprop(measurement_model.parameters(), lr=learning_rate)
  elif args.optimizer == 'sgd':
      optimizer = optim.SGD(measurement_model.parameters(), lr=learning_rate)
  else:
      raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

  # Creating folders to log
  run_name = "Diff_HEF_R2"
  current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = os.path.join(base_path,"logs", run_name, current_datetime)
  os.makedirs(log_dir, exist_ok=True)

  hed = HarmonicExponentialDistribution(range_x,range_y,band_limit,step_t)
  hef = HEFilter(band_limit,range_x,range_y)
  batch_step = 0
  for epoch in range(num_epochs):
    loss_tot = 0
    kl_2_tot = 0
    true_nll_tot = 0
    mae_tot = 0
    rmse_tot = 0
    measurement_nll_tot = 0
    val_dict = validate_model(measurement_model, val_loader,  hed, band_limit, range_x, range_y, measurement_noise, initial_noise, hef,trajectory_length,step_t,motion_noise)
    val_dict['Epoch'] = epoch + 1
    
    if args.decay_lr and epoch % 10  == 0:
      learning_rate = learning_rate * (0.95 ** epoch)
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    wandb.log(val_dict)
    print(f"Training Epoch: {epoch + 1}")
    for batch_idx, (ground_truth, measurements) in enumerate(train_loader):
        prior = GaussianDistribution(ground_truth[:, 0], initial_noise, range_x, range_y,band_limit)
        prior_pdf = prior.density_over_grid()
        analytic_prior  = prior_pdf.clone()
        control = torch.ones((batch_size,2)) * torch.tensor(step_t)  + np.random.normal(loc=0, scale=motion_noise, size=(batch_size,2))
        for j in range(trajectory_length):      
            input_measurement = GaussianDistribution(measurements[:, j],initial_noise,range_x,range_y,band_limit)
            input_measurement_pdf = input_measurement.density_over_grid()
            process = GaussianDistribution(control, motion_noise, range_x, range_y,band_limit)
            process_pdf  = process.density_over_grid()
            prior_pdf = prior_pdf.detach()
            # Predict Step
            eta_bel_x_t_bar, density_bel_x_t_bar = hef.predict(prior_pdf, process_pdf)
            density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32)
            # P(z_t | x_t) Measurement Step
            measurement_pdf = measurement_model(input_measurement_pdf,density_bel_x_t_bar)
            z = hed.normalization_constant(measurement_pdf)
            measurement_pdf = measurement_pdf/z
            # Calculate Posterior
            posterior_pdf = hef.update(eta_bel_x_t_bar, measurement_pdf)
            posterior_pdf = posterior_pdf.to(torch.float32)
            loss = hed.negative_log_likelihood_density(posterior_pdf,ground_truth[:,j])
            target_distribution = GaussianDistribution(measurements[:,j],measurement_noise,range_x,range_y,band_limit)
            true_nll_input = target_distribution.negative_log_likelihood(measurements)
            true_density = target_distribution.density_over_grid()
            measurement_nll = hed.negative_log_likelihood_density(measurement_pdf,measurements[:,j])

            mode = hed.mode(posterior_pdf)
            rmse_tot += root_mean_square_error(mode,ground_truth[:,j]).item()
            mae_tot += mean_absolute_error(mode,ground_truth[:,j]).item()
            kl_div_2 = kl_divergence(true_density,measurement_pdf)
        
            optimizer.zero_grad()
            loss.backward()
            for name, param in measurement_model.named_parameters():
                if param.grad is not None:
                    wandb.log({'batch': batch_step,f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
            optimizer.step()
            
            loss_tot +=loss.item()
            kl_2_tot += kl_div_2.item()
            true_nll_tot += true_nll_input.item()
            measurement_nll_tot += measurement_nll.item()
    
            batch_step += 1
            analytic_posterior, analytic_measurement, analytic_predict = hef.analytic_filter(control,motion_noise,measurements[:,j],measurement_noise,analytic_prior)
        
            if batch_idx==0 and epoch % 20 == 0:
                epoch_dir = log_dir + f"/epoch_{epoch}"
                os.makedirs(epoch_dir, exist_ok=True)
                plot_3d_filter(ground_truth[sample_idx,j],measurements[sample_idx,j],range_x,range_y,band_limit, true_density[sample_idx],measurement_pdf[sample_idx],density_bel_x_t_bar[sample_idx],posterior_pdf[sample_idx],epoch_dir,traj_iter=j)
                plot_3d(ground_truth[sample_idx,j],measurements[sample_idx,j],range_x,range_y,band_limit, analytic_posterior[sample_idx],analytic_measurement[sample_idx],analytic_predict[sample_idx],epoch_dir,j)
                # wandb.log({rf"plot3d_{epoch}": wandb.Image(epoch_dir + "/plot3d.png")})
                # wandb.log({rf"plot2d_{epoch}": wandb.Image(epoch_dir + "/2d_plots.png")})
            # Current posterior is prior for the next step (j + 1)
            prior_pdf = posterior_pdf 
            analytic_prior = analytic_posterior
        if batch_idx==0 and epoch % 10 == 0:
           if os.path.exists(epoch_dir):
            epoch_dir = str(epoch_dir)
            video_path_1 = str(log_dir + f"/video_{epoch}_plot3d.gif")
            generate_gif(epoch_dir,video_path_1,prefix='plot3d',duration=2)
            wandb.log({rf"video_{epoch}": wandb.Video(video_path_1)})
            video_path_2 = str(log_dir + f"/video_{epoch}_2d_plots.gif")
            generate_gif(epoch_dir,video_path_2,prefix='2d_plots',duration=2)
            wandb.log({rf"video_{epoch}": wandb.Video(video_path_2)})
            video_path_3 = str(log_dir + f"/video_{epoch}_hef_2d_plots.gif")
            generate_gif(epoch_dir,video_path_3,prefix='hef_2d_plots',duration=2)
            wandb.log({rf"video_{epoch}": wandb.Video(video_path_3)})
    wandb.log({
              'Epoch': epoch + 1,
              'Train Posterior loss': loss_tot / (len(train_loader)*trajectory_length),
              'Train KL Div' : kl_2_tot /(len(train_loader)*trajectory_length),   
              'Train MAE' : mae_tot/(len(train_loader)*trajectory_length),
              'Train RMSE' : rmse_tot/(len(train_loader)*trajectory_length),
              'True NLL' : true_nll_tot/(len(train_loader)*trajectory_length),
              'Measurement NLL' : measurement_nll_tot/(len(train_loader)*trajectory_length)
              })
  print("Training Completed")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='CNN Uncertainty Estimation in R2')
  parser.add_argument('--n_samples', type=int, default=800, help='Number of samples')
  parser.add_argument('--trajectory_length', type=int, default=30, help='Trajectory length')
  parser.add_argument('--measurement_noise', type=float, default=0.1, help='Measurement noise')
  parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
  parser.add_argument('--initial_noise', type=float, default=0.2, help='Initial noise')
  parser.add_argument('--band_limit', type=str, default="50 50", help='Band limit')
  parser.add_argument('--range_x_start', type=float, default=-0.5, help='Range x start')
  parser.add_argument('--range_x_end', type=float, default=0.5, help='Range x end')
  parser.add_argument('--range_y_start', type=float, default=-0.5, help='Range y start')
  parser.add_argument('--range_y_end', type=float, default=0.5, help='Range y end')
  parser.add_argument('--optimizer', type=str, default='rmsprop', help='Optimizer type (adam, rmsprop, sgd)')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--decay_lr', type=int, default=0, help='Decay learning rate')
  parser.add_argument('--motion_noise', type=float, default=0.05, help='Motion noise')

  args = parser.parse_args()
  band_limit = [int(x) for x in args.band_limit.split()]
  args.band_limit = band_limit
  torch.manual_seed(args.seed)

    # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)

  wandb.init(mode='disabled',project="Diff-HEF",group="R2",entity="korra141",
              tags=["R2", "Filter","CNN","Learning Measurement Model","Analytic Process Model"],
              name="R2-DiffHEF",
              config=args)
  main(args)