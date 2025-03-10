
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

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

# local import 
from src.utils.visualisation import plot_density,fit_grid_into_larger
from src.distributions.R2.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.R2.StandardDistribution import GaussianDistribution,MultiModalGaussianDistribution
from src.utils.debug_tools import check_model_weights_nan, check_tensor_nan, get_gradients
from src.data_generation.R2.toy_dataset import TrajectoryGenerator
from src.models.CNN import DensityPredictorCNN, CNNModel_A, initialize_weights,init_weights
from src.utils.metrics import kl_divergence, mean_absolute_error, root_mean_square_error
from src.models.MLP import NeuralNetwork,init_weights_mlp

def validate_model(model, data_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise,delta_flag):
  model.eval()
  val_dict = {}
  with torch.no_grad():
    val_loss_tot = 0
    val_kl_2_tot = 0
    val_true_nll_tot = 0
    for val_batch_idx, (val_ground_truth, val_measurements) in enumerate(data_loader):
      print(f"Validation Batch: {val_batch_idx}")
      input_pose_pdf = GaussianDistribution(val_ground_truth,initial_noise,range_x,range_y,band_limit)
      target_distribution = GaussianDistribution(val_ground_truth,measurement_noise,range_x,range_y,band_limit)

      val_input_pose_density = input_pose_pdf.density_over_grid()
      val_outputs = model(val_input_pose_density)
      val_z = hed.normalization_constant(val_outputs)
      val_outputs = val_outputs / val_z
      if delta_flag:
         predicted_density = hed.convolve(val_input_pose_density,val_outputs)
      else:
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
  n_modes = args.n_modes
  measurement_noise = torch.ones(n_modes) * args.measurement_noise
  initial_noise = args.initial_noise
  range_x = (args.range_x_start, args.range_x_end)
  range_y = (args.range_y_start, args.range_y_end)
  band_limit = args.band_limit
  scaling_factor = args.scaling_factor
  mean_offset = args.mean_offset
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  num_epochs = args.num_epochs


  range_x_diff = range_x[1] - range_x[0]
  range_y_diff = range_y[1] - range_y[0]
  range_x_diff_local = args.range_x_diff_local
  range_y_diff_local = args.range_y_diff_local
  step_t = (round(range_x_diff_local/band_limit[0], 2), round(range_y_diff_local/band_limit[1], 2))
  # Scheduler for regularization parameter lambda_
  lambda_scheduler = lambda epoch: args.start_lambda + (args.end_lambda - args.start_lambda) * (1 - torch.exp(torch.tensor(-args.reg_factor * epoch / num_epochs)))
  sample_idx = 0

  centers = torch.tile(torch.linspace(-mean_offset / 2, mean_offset / 2, n_modes)[None,:,None], (batch_size, 1, 2)) # n_modes


  data_path = os.path.join(base_path, 'data')
  os.makedirs(data_path, exist_ok=True)
  data_generator = TrajectoryGenerator(range_x, range_y, step_t, n_samples , trajectory_length, measurement_noise, mean_offset, n_modes)
  train_loader, val_loader = data_generator.create_data_loaders(data_path, batch_size,flag_flattend=True)

  if args.architecture == 0:
    print("Using CNN Model with batch norm kaiming uniform")
    model = CNNModel_A(band_limit)
    model.apply(initialize_weights)
  elif args.architecture == 1:
    print("Using CNN Model with batch norm kaiming norm")
    model = CNNModel_A(band_limit)
    model.apply(init_weights)
  elif args.architecture == 2:
    print("Using CNN Model with kaiming uniform")
    model = DensityPredictorCNN(1,band_limit,batch_size)
    model.apply(initialize_weights)
  elif args.architecture == 3:
    print("Using CNN Model with kaiming norm")
    model = DensityPredictorCNN(1,band_limit,batch_size)
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
  run_name = "MM UE in R2 Local Grid"
  current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = os.path.join(base_path,"logs", run_name, current_datetime)
  os.makedirs(log_dir, exist_ok=True)

  hed = HarmonicExponentialDistribution(band_limit,step_t,range_x_diff=range_x_diff_local,range_y_diff=range_y_diff_local)
  batch_step = 0
  for epoch in range(num_epochs):
    loss_tot = 0
    kl_2_tot = 0
    true_nll_tot = 0
    mae_tot = 0
    rmse_tot = 0
    # val_dict = validate_model(model, val_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise,args.delta)
    # val_dict['Epoch'] = epoch + 1
    
    if args.decay_lr:
      learning_rate_decay = args.learning_rate_start + (args.learning_rate_end - args.learning_rate_start) * (1 - np.exp((-args.lr_factor * epoch / num_epochs)))
      print("learning_rate",learning_rate_decay)
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_decay
    lambda_ = lambda_scheduler(epoch)
    print("regularisation factor",lambda_)
    # wandb.log(val_dict)
    print(f"Training Epoch: {epoch + 1}")
    for batch_idx, (ground_truth, measurements) in enumerate(train_loader):
      check_model_weights_nan(model)
      input_pose_pdf = GaussianDistribution(ground_truth, initial_noise, band_limit, range_x_diff=range_x_diff_local, range_y_diff=range_y_diff_local)
      input_pose_density = input_pose_pdf.density_over_local_grid()
      outputs = model(input_pose_density)
      check_tensor_nan(outputs)
      z = hed.normalization_constant(outputs)
      check_tensor_nan(z)
      outputs = outputs/z
      predicted_density = outputs
      # loss = torch.mean(hed.negative_log_likelihood_local_grid(predicted_density, measurements,ground_truth))
      loss = hed.loss_regularisation_norm(lambda_,predicted_density, measurements,ground_truth,scaling_factor)
      mm_mean = ground_truth.unsqueeze(1) + centers
      target_distribution = MultiModalGaussianDistribution(mm_mean,measurement_noise,band_limit,n_modes,x_range=range_x,y_range=range_y)
      true_nll_input = target_distribution.negative_log_likelihood(measurements)
      true_density = target_distribution.density_over_grid()
      target_distribution_local_grid = MultiModalGaussianDistribution(mm_mean,measurement_noise,band_limit,n_modes,range_x_diff=range_x_diff_local,range_y_diff=range_y_diff_local)
      true_density_local_grid = target_distribution_local_grid.density_over_local_grid()

      # mode = hed.mode(predicted_density) + ground_truth
      # rmse_tot += root_mean_square_error(mode,ground_truth).item()
      # mae_tot += mean_absolute_error(mode,ground_truth).item()
      kl_div_2 = kl_divergence(true_density_local_grid, predicted_density)
   
      optimizer.zero_grad()
      loss.backward()
      if batch_idx % 100 == 0:
        for name, param in model.named_parameters():
          if param.grad is not None:
            wandb.log({'batch': batch_step,f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
      optimizer.step()
      loss_tot +=loss.item()
      kl_2_tot += kl_div_2.item()
      true_nll_tot += true_nll_input.item()
    
      wandb.log({'batch': batch_step, 'Train batch NL Loss': loss.item(), 'Train batch KL Div 2': kl_div_2.item()})
      batch_step += 1
      
      if batch_idx==0 and epoch % 10 == 0:
        plot_dict = {
          'true_energy': true_density[sample_idx],
          'predicted_energy': fit_grid_into_larger(ground_truth[sample_idx],range_x_diff_local,range_y_diff_local,band_limit,predicted_density[sample_idx]),
        }
        plot_density(ground_truth[sample_idx],measurements[sample_idx],range_x,range_y,log_dir,epoch,plot_dict,"CNN MM Local UE density")
    wandb.log({
              'Epoch': epoch + 1,
              'Train NLL loss': loss_tot / len(train_loader),
              'Train KL Div 2' : kl_2_tot / len(train_loader),   
              'True NLL' : true_nll_tot/len(train_loader),
              })
  print("Training Completed")
  for img_file in os.listdir(log_dir):
      if img_file.endswith(".png"):
        wandb.log({img_file: wandb.Image(os.path.join(log_dir, img_file))})
  wandb.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='CNN Uncertainty Estimation in R2')
  parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
  parser.add_argument('--trajectory_length', type=int, default=100, help='Trajectory length')
  parser.add_argument('--measurement_noise', type=float, default=0.1, help='Measurement noise')
  parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
  parser.add_argument('--initial_noise', type=float, default=0.05, help='Initial noise')
  parser.add_argument('--band_limit', type=str, default="30 30", help='Band limit')
  parser.add_argument('--range_x_start', type=float, default=-0.5, help='Range x start')
  parser.add_argument('--range_x_end', type=float, default=0.5, help='Range x end')
  parser.add_argument('--range_y_start', type=float, default=-0.5, help='Range y start')
  parser.add_argument('--range_y_end', type=float, default=0.5, help='Range y end')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (adam, rmsprop, sgd)')
  parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--decay_lr', type=int, default=0, help='Decay learning rate')
  parser.add_argument('--range_x_diff_local', type=float, default=0.5, help='Local range x difference')
  parser.add_argument('--range_y_diff_local', type=float, default=0.5, help='Local range y difference')
  parser.add_argument('--lambda_', type=float, default=0, help='Regularization parameter lambda')
  parser.add_argument('--scaling_factor', type=float, default=1.0, help='Scaling factor for regularization')
  parser.add_argument('--mean_offset', type=float, default=0.15, help='Mean offset for multimodal distribution')
  parser.add_argument('--n_modes', type=int, default=2, help='Number of modes for multimodal distribution')
  parser.add_argument('--architecture', type=int, choices=[0, 1, 2, 3], default=0, help='Choice of architecture (0, 1, 2, 3, 4)')
  parser.add_argument('--start_lambda', type=float, default=0.1, help='Starting value of lambda for regularization')
  parser.add_argument('--end_lambda', type=float, default=100, help='Ending value of lambda for regularization')
  parser.add_argument('--reg_factor', type=float, default=5, help='Regularization factor for lambda scheduler')
  parser.add_argument('--learning_rate_start', type=float, default=0.01, help='Starting learning rate for decay')
  parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Ending learning rate for decay')
  parser.add_argument('--lr_factor', type=float, default=10, help='Factor for learning rate adjustment')
  args = parser.parse_args()
  band_limit = [int(x) for x in args.band_limit.split()]
  args.band_limit = band_limit
  torch.manual_seed(args.seed)

    # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)

  wandb.init(project="Diff-HEF",group="R2",entity="korra141",
              tags=["R2", "Uncertainity Estimation", "CNN", "predicting density","predicting delta density","measurement likelihood","local grid","unimodal"],
              name="R2-LocalGridUncertainityEstimation",
              config=args)
  main(args)