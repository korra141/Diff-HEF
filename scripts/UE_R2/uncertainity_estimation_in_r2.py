import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import datetime
import argparse
import wandb
import sys
import torch.optim as optim
import pdb


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

# local import 
from src.utils.visualisation import plot_3d
from src.distributions.R2.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from distributions.R2.StandardDistribution import GaussianDistribution
from src.utils.debug_tools import check_model_weights_nan, check_tensor_nan, get_gradients
from src.data_generation.R2.toy_dataset import create_data_loaders
from src.models.MLP import NeuralNetwork,init_weights,init_weights_one
from src.utils.metrics import kl_divergence, mean_absolute_error, root_mean_square_error

def validate_model(model, data_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise):
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
      predicted_density = hed.convolve(val_input_pose_density,val_outputs )
      val_loss = hed.negative_log_likelihood_density(predicted_density, val_measurements)
      val_true_nll_input = target_distribution.negative_log_likelihood(val_measurements)
      val_true_density = target_distribution.density_over_grid()

      val_kl_div_2 = kl_divergence(val_true_density, val_outputs)

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
  # Define hyperparameters
  n_samples = args.n_samples
  trajectory_length = args.trajectory_length
  measurement_noise = args.measurement_noise
  batch_size = args.batch_size
  initial_noise = args.initial_noise
  band_limit = args.band_limit
  range_x = args.range_x
  range_y = args.range_y
  learning_rate = args.learning_rate
  num_epochs = args.num_epochs

  range_x_diff = range_x[1] - range_x[0]
  range_y_diff = range_y[1] - range_y[0]
  step_t = (round(range_x_diff/band_limit[0], 2), round(range_y_diff/band_limit[1], 2))
  sample_idx = 0

  data_path = os.path.join(base_path, 'data')
  train_loader, val_loader = create_data_loaders(data_path, range_x, range_y, step_t, n_samples, trajectory_length, measurement_noise, batch_size)

  # Initialize the model and optimizer
  model = NeuralNetwork(np.prod(band_limit),band_limit,batch_size) # Assuming input and output dimensions are 3
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
  run_name = "Uncertainity Estimation in R2"
  current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = os.path.join("logs", run_name, current_datetime)
  os.makedirs(log_dir, exist_ok=True)

  hed = HarmonicExponentialDistribution(range_x,range_y,band_limit,step_t)
  batch_step = 0
  # Training loop
  for epoch in range(num_epochs):
    loss_tot = 0
    kl_2_tot = 0
    true_nll_tot = 0
    hef_loss_temp = 0
    initial_nll_temp = 0
    mae_tot = 0
    rmse_tot = 0
    val_dict = validate_model(model, val_loader, hed, band_limit, range_x, range_y, measurement_noise, initial_noise)
    val_dict['Epoch'] = epoch + 1
    if args.decay_lr and epoch % 10  == 0:
      learning_rate = learning_rate * (0.95 ** epoch)
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    wandb.log(val_dict)
    print(f"Training Epoch: {epoch + 1}")
    for batch_idx, (ground_truth, measurements) in enumerate(train_loader):

      check_model_weights_nan(model)
      input_pose_pdf = GaussianDistribution(ground_truth,initial_noise,range_x,range_y,band_limit)
      input_pose_density = input_pose_pdf.density_over_grid()
      outputs = model(input_pose_density)
      check_tensor_nan(outputs)
      z = hed.normalization_constant(outputs)
      check_tensor_nan(z)
      outputs = outputs/z
      predicted_density = hed.convolve(input_pose_density,outputs)
      hef_loss_input = hed.negative_log_likelihood_density(input_pose_density,measurements)
      initial_nll = input_pose_pdf.negative_log_likelihood(measurements)
    
      loss = hed.negative_log_likelihood_density(predicted_density, measurements)

      target_distribution = GaussianDistribution(ground_truth,measurement_noise,range_x,range_y,band_limit)
      true_nll_input = target_distribution.negative_log_likelihood(measurements)
      true_density = target_distribution.density_over_grid()

      mode = hed.mode(predicted_density)
      rmse_tot += root_mean_square_error(mode,ground_truth).item()
      mae_tot += mean_absolute_error(mode,ground_truth).item()
      kl_div_2 = kl_divergence(true_density,predicted_density)
   
      optimizer.zero_grad()
      loss.backward()
      for name, param in model.named_parameters():
        if param.grad is not None:
          wandb.log({'batch': batch_step,f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
      optimizer.step()
      loss_tot +=loss.item()
      kl_2_tot += kl_div_2.item()
      true_nll_tot += true_nll_input.item()
      initial_nll_temp += initial_nll.item()
      hef_loss_temp +=hef_loss_input.item()
    
      wandb.log({'batch': batch_step, 'Train batch NL Loss': loss.item(), 'Train batch KL Div 2': kl_div_2.item()})
      batch_step += 1
      
      if batch_idx==0 and epoch % 100 == 0:
        epoch_dir = log_dir + f"/epoch_{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)
        plot_3d(ground_truth[sample_idx],measurements[sample_idx],range_x,range_y,band_limit,true_density[sample_idx],predicted_density[sample_idx],input_pose_density[sample_idx],epoch_dir,outputs[sample_idx])
        wandb.log({rf"plot3d_{epoch}": wandb.Image(epoch_dir + "/plot3d.png")})
        wandb.log({rf"plot2d_{epoch}": wandb.Image(epoch_dir + "/2d_plots.png")})

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
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Uncertainty Estimation in R2')
  parser.add_argument('--n_samples', type=int, default=500, help='Number of samples')
  parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
  parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
  parser.add_argument('--measurement_noise', type=float, default=0.2, help='Measurement noise')
  parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--initial_noise', type=float, default=0.05, help='Initial noise')
  parser.add_argument('--band_limit', type=str, default="100 100",help="Bandlimit values as a string")
  parser.add_argument('--range_x', type=float, nargs=2, default=[-0.5, 0.5], help='Range for x')
  parser.add_argument('--range_y', type=float, nargs=2, default=[-0.5, 0.5], help='Range for y')
  parser.add_argument('--seed', type=int, default=12345, help='Range for y')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
  parser.add_argument('--decay_lr', type=int, default=0, help='Decay learning rate')
  parser.add_argument('--delta', type=int, default=0, choices=[0,1], help='Predict delta density')

  
  args = parser.parse_args()

  band_limit = [int(x) for x in args.band_limit.split()]
  args.band_limit = band_limit
  torch.manual_seed(args.seed)

    # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)

  wandb.init(project="Diff-HEF",group="R2",entity="korra141",
              tags=["R2", "Uncertainity Estimation", "MLP", "predicting delta density","measurement likelihood"],
              name="R2-UncertainityEstimation",
              config=args)
  main(args)