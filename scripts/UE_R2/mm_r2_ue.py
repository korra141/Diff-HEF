
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
from src.utils.visualisation import plot_3d,plot_3d_density
from src.distributions.R2.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.R2.StandardDistribution import MultiModalGaussianDistribution, GaussianDistribution
from src.utils.debug_tools import check_model_weights_nan, check_tensor_nan, get_gradients
from src.data_generation.R2.toy_dataset import TrajectoryGenerator
from src.models.CNN import CNNModel_A,init_weights
from src.utils.metrics import kl_divergence, mean_absolute_error, root_mean_square_error

def validate_model(model, data_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise,delta_flag,centers,n_modes):
  model.eval()
  val_dict = {}
  with torch.no_grad():
    val_loss_tot = 0
    val_kl_2_tot = 0
    val_true_nll_tot = 0
    for val_batch_idx, (val_ground_truth, val_measurements) in enumerate(data_loader):
      print(f"Validation Batch: {val_batch_idx}")
      input_pose_pdf = GaussianDistribution(val_ground_truth,initial_noise,range_x,range_y,band_limit)
      mm_mean = val_measurements.unsqueeze(1) + centers
      target_distribution = MultiModalGaussianDistribution(mm_mean,measurement_noise,range_x,range_y,band_limit,n_modes)

      val_input_pose_density = input_pose_pdf.density_over_grid()
      val_outputs = model(val_input_pose_density)
      val_z = hed.normalization_constant(val_outputs)
      val_outputs = val_outputs / val_z
      predicted_density = val_outputs
      val_loss = hed.negative_log_likelihood_density(predicted_density, val_measurements)
      val_true_nll_input = target_distribution.negative_log_likelihood(val_measurements)
      val_true_density = target_distribution.density_over_grid()

      val_kl_div_2 = kl_divergence(val_true_density, predicted_density)

      val_loss_tot += val_loss.item()
      val_kl_2_tot += val_kl_div_2.item()
      val_true_nll_tot += val_true_nll_input.item()
    
    val_dict= {
              'Val NLL loss': val_loss_tot / len(data_loader),
              'Val KL Div 2' : val_kl_2_tot / len(data_loader),
              'Val True NLL' : val_true_nll_tot/len(data_loader)
              }
    return val_dict
  
def main(args):
  # Data Parameters
  n_samples = args.n_samples
  trajectory_length = args.trajectory_length
  
  range_x = (args.range_x_start, args.range_x_end)
  range_y = (args.range_y_start, args.range_y_end)
  band_limit = args.band_limit
  n_modes = args.n_modes
  measurement_noise= torch.ones(n_modes) * args.measurement_noise
  initial_noise = args.initial_noise
  mean_offset = args.mean_offset

  range_x_diff = range_x[1] - range_x[0]
  range_y_diff = range_y[1] - range_y[0]
  step_t = (round(range_x_diff/band_limit[0], 2), round(range_y_diff/band_limit[1], 2))
  sample_idx = 0

  # Training Parameters
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  num_epochs = args.num_epochs

  data_path = os.path.join(base_path, 'data')
  data_generator = TrajectoryGenerator(range_x, range_y, step_t, n_samples , trajectory_length, measurement_noise, mean_offset, n_modes)
  train_loader, val_loader = data_generator.create_data_loaders(data_path, batch_size,flag_flattend=True)

  model = CNNModel_A(band_limit) # Assuming input and output dimensions are 3
  model.apply(init_weights)

  if args.optimizer == 'adam':
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  elif args.optimizer == 'rmsprop':
      optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
  elif args.optimizer == 'sgd':
      optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  else:
      raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

  # Creating folders to log
  run_name = " MM Uncertainity Estimation in R2"
  current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = os.path.join(base_path,"logs", run_name, current_datetime)
  os.makedirs(log_dir, exist_ok=True)

  centers = torch.tile(torch.linspace(-mean_offset / 2, mean_offset / 2, n_modes)[None,:,None], (batch_size, 1, 2)) # n_modes
      
  hed = HarmonicExponentialDistribution(range_x,range_y,band_limit,step_t)
  batch_step = 0
  for epoch in range(num_epochs):
    loss_tot = 0
    kl_2_tot = 0
    true_nll_tot = 0
    hef_loss_temp = 0
    initial_nll_temp = 0
    mae_tot = 0
    rmse_tot = 0
    # val_dict = validate_model(model, val_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise,args.delta,centers,n_modes)
    # val_dict['Epoch'] = epoch + 1
    
    if args.decay_lr and epoch % 1  == 0:
      learning_rate_decay = learning_rate * (0.8 ** epoch)
      print(learning_rate_decay)
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_decay
    # wandb.log(val_dict)
    print(f"Training Epoch: {epoch + 1}")
    for batch_idx, (ground_truth, measurements) in enumerate(train_loader):
      start_time = datetime.datetime.now()
      check_model_weights_nan(model)
      input_pose_pdf = GaussianDistribution(ground_truth,initial_noise,range_x,range_y,band_limit)
      input_pose_density = input_pose_pdf.density_over_grid()
      outputs = model(input_pose_density)
      check_tensor_nan(outputs)
      z = hed.normalization_constant(outputs)
      check_tensor_nan(z)
      outputs = outputs/z
      predicted_density = outputs
      hef_loss_input = hed.negative_log_likelihood_density(input_pose_density,measurements)
      initial_nll = input_pose_pdf.negative_log_likelihood(measurements)
    
      # loss = hed.negative_log_likelihood_density(predicted_density, measurements)
      loss = hed.loss_energy(predicted_density, measurements)
      mm_mean = ground_truth.unsqueeze(1) + centers #(batch_size, n_modes, 2)
      target_distribution = MultiModalGaussianDistribution(mm_mean,measurement_noise,range_x,range_y,band_limit,n_modes)
      true_nll_input = target_distribution.negative_log_likelihood(measurements)
      true_density = target_distribution.density_over_grid()

      mode = hed.mode(predicted_density)
      rmse_tot += root_mean_square_error(mode,measurements).item()
      mae_tot += mean_absolute_error(mode,measurements).item()
      kl_div_2 = kl_divergence(true_density,predicted_density)
   
      optimizer.zero_grad()
      loss.backward()
      # for name, param in model.named_parameters():
      #   if param.grad is not None:
      #     wandb.log({'batch': batch_step,f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
      optimizer.step()
      loss_tot +=loss.item()
      kl_2_tot += kl_div_2.item()
      true_nll_tot += true_nll_input.item()
      initial_nll_temp += initial_nll.item()
      hef_loss_temp +=hef_loss_input.item()
      # print(f'batch: {batch_step}, Train batch NL Loss: {loss.item()}, Train batch KL Div 2: {kl_div_2.item()}, time: {datetime.datetime.now() - start_time}')
    
      # wandb.log({'batch': batch_step, 'Train batch NL Loss': loss.item(), 'Train batch KL Div 2': kl_div_2.item()})
      batch_step += 1
      
      if batch_idx==0 and epoch % 10 == 0:
        epoch_dir = log_dir + f"/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        dict_density = {"true_density": true_density[sample_idx], "predicted_density": predicted_density[sample_idx], "input_density": input_pose_density[sample_idx]}
        plot_3d_density(ground_truth[sample_idx],measurements[sample_idx],range_x,range_y,band_limit,epoch_dir, dict_density,"Uncertainity Estimation in R2 MM")
    wandb.log({
              'Epoch': epoch + 1,
              'Train NLL loss': loss_tot / len(train_loader),
              'Train KL Div 2' : kl_2_tot / len(train_loader),   
              'Train MAE' : mae_tot/len(train_loader),
              'Train RMSE' : rmse_tot/len(train_loader),
              'True NLL' : true_nll_tot/len(train_loader),
              'Input NLL' : initial_nll_temp/len(train_loader),
              'IUR Input NLL' : hef_loss_temp/len(train_loader)
              })
  print("Training Completed")
  for epoch_folder in os.listdir(log_dir):
    epoch_folder_path = os.path.join(log_dir, epoch_folder)
    if os.path.isdir(epoch_folder_path):
      wandb.log({rf"plot3d_{epoch}": wandb.Image(os.path.join(epoch_folder_path, "plot3d.png"))})
      wandb.log({rf"plot2d_{epoch}": wandb.Image(os.path.join(epoch_folder_path, "2d_plots.png"))})
  wandb.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='CNN Uncertainty Estimation in R2')
  parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
  parser.add_argument('--trajectory_length', type=int, default=100, help='Trajectory length')
  parser.add_argument('--measurement_noise', type=float, default=0.05, help='Measurement noise')
  parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
  parser.add_argument('--initial_noise', type=float, default=0.2, help='Initial noise')
  parser.add_argument('--band_limit', type=str, default="50 50", help='Band limit')
  parser.add_argument('--range_x_start', type=float, default=-0.5, help='Range x start')
  parser.add_argument('--range_x_end', type=float, default=0.5, help='Range x end')
  parser.add_argument('--range_y_start', type=float, default=-0.5, help='Range y start')
  parser.add_argument('--range_y_end', type=float, default=0.5, help='Range y end')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (adam, rmsprop, sgd)')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--decay_lr', type=int, default=1, help='Decay learning rate')
  parser.add_argument('--delta', type=int, default=0, help='output delta density')
  parser.add_argument('--n_modes', type=int, default=2, help='Number of modes for the Gaussian distribution')
  parser.add_argument('--mean_offset', type=float, default=0.2, help='Mean offset for the Gaussian distribution')

  args = parser.parse_args()
  band_limit = [int(x) for x in args.band_limit.split()]
  args.band_limit = band_limit
  torch.manual_seed(args.seed)

    # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)

  wandb.init(project="Diff-HEF",group="R2",entity="korra141",
              tags=["R2", "Uncertainity Estimation", "CNN", "predicting density","MultiModal","ModelInitialization"],
              name="R2-UncertainityEstimation",
              config=args)
  main(args)