
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
from src.distributions.R2.StandardDistribution import GaussianDistribution
from src.utils.debug_tools import check_model_weights_nan, check_tensor_nan, get_gradients
from src.data_generation.R2.toy_dataset import TrajectoryGenerator
from src.models.CNN import DensityPredictorMLPCNN, DensityPredictorCNN, CNNModel_A, init_weights_identity,initialize_weights
from src.models.MLP import NeuralNetwork,init_weights_mlp, SimpleModel
from src.utils.metrics import kl_divergence, mean_absolute_error, root_mean_square_error

def validate_model(model, data_loader, hed, args ,localgrid,epoch,log_dir,range_x, range_y):
  model.eval()
  log_freq = len(data_loader)/3
  true_cov = torch.tensor([args.measurement_noise**2,args.measurement_noise**2]).unsqueeze(0)
  initial_cov = torch.tensor([args.initial_noise**2,args.initial_noise**2]).unsqueeze(0)
  with torch.no_grad():
    rmse_tot = 0
    mae_tot = 0
    kl_2_tot = 0
    kl_1_tot = 0
    true_nll_tot = 0
    nll_tot = 0
    rmse_cov_tot = 0
    for batch_idx, (ground_truth, measurements) in enumerate(data_loader):
      if localgrid:
          input_pose_pdf = GaussianDistribution(ground_truth, initial_cov,  args.band_limit, range_x_diff=args.range_x_diff_local, range_y_diff=args.range_y_diff_local)
          input_pose_density = input_pose_pdf.density_over_local_grid()
      else:
        input_pose_pdf = GaussianDistribution(ground_truth,initial_cov,args.band_limit, range_x,range_y)
        input_pose_density = input_pose_pdf.density_over_grid()
      if args.gaussian_param:
        mu_x,mu_y,logcov_x, logcov_y = model(ground_truth)
        mu = torch.cat((mu_x,mu_y),dim=1)
        cov = torch.cat((torch.exp(logcov_x),torch.exp(logcov_y)),dim=1)
        mu_ = mu.clone().detach()
        cov_ = cov.clone().detach()
        if localgrid:
          predicted_distribution = GaussianDistribution(mu,cov,args.band_limit,range_x_diff=args.range_x_diff_local,range_y_diff=args.range_y_diff_local)
          predicted_density = predicted_distribution.density_over_local_grid()
        else:
          predicted_distribution = GaussianDistribution(mu,cov,args.band_limit,x_range=range_x,y_range=range_y)
          predicted_density = predicted_distribution.density_over_grid()
      else: 
        if args.architecture == 0:
          outputs = model(ground_truth)
        elif args.architecture == 1:
          outputs = model(input_pose_density)
        elif args.architecture == 2:
          outputs = model(input_pose_density)
        elif args.architecture == 3:
          outputs = model(ground_truth)
          # hed.convolve(input_pose_density,outputs)
        else:
          raise ValueError(f"Unsupported architecture type: {args.architecture}")
        check_tensor_nan(outputs)
        z = hed.normalization_constant(outputs)
        check_tensor_nan(z)
        outputs = outputs/z
        predicted_density = outputs
      target_distribution = GaussianDistribution(ground_truth,true_cov,args.band_limit,x_range=range_x,y_range=range_y)
      true_nll_input = target_distribution.negative_log_likelihood(measurements)
      true_density_plot = target_distribution.density_over_grid()

      if args.gaussian_nll and args.gaussian_param:
        nll = predicted_distribution.negative_log_likelihood(measurements)
        rmse_tot += root_mean_square_error(mu_,ground_truth).item()
        mae_tot += mean_absolute_error(mu_,ground_truth).item()
        measurement_cov  = torch.ones_like(cov_)* (args.measurement_noise**2)
        rmse_cov_tot += root_mean_square_error(cov_,measurement_cov).item()
      else:
        if localgrid:
          # loss = hed.loss_regularisation_norm(lambda_,predicted_density, measurements,mean=ground_truth,scaling_factor=scaling_factor)
          nll = torch.mean(hed.negative_log_likelihood_local_grid(predicted_density, measurements,mean=ground_truth))
          target_distribution_local_grid = GaussianDistribution(ground_truth,true_cov,args.band_limit,range_x_diff=args.range_x_diff_local,range_y_diff=args.range_y_diff_local)
          true_density_metric = target_distribution_local_grid.density_over_local_grid()
          mode = hed.mode(predicted_density, ground_truth)
        else:
          # loss = hed.loss_regularisation_norm(lambda_,predicted_density, measurements,scaling_factor=scaling_factor)
          nll = torch.mean(hed.loss_energy(predicted_density, measurements))
          true_density_metric = true_density_plot
          mode = hed.mode(predicted_density)
        rmse_tot += root_mean_square_error(mode,ground_truth).item()
        mae_tot += mean_absolute_error(mode,ground_truth).item()
      kl_div_2 = kl_divergence(true_density_metric , predicted_density)
      kl_div_1 = kl_divergence(predicted_density, true_density_metric )
      kl_2_tot += kl_div_2.item()
      kl_1_tot += kl_div_1.item()
      true_nll_tot += true_nll_input.item()
      nll_tot += nll.item()
      if epoch % 50 == 0 and batch_idx == 0:
        indices = np.random.choice(args.batch_size, 5, replace=False)
        for j in indices:
          plot_dict = {}
          if localgrid:
            plot_dict = {
              'true_energy': true_density_plot[j],
              'predicted_energy': fit_grid_into_larger(ground_truth[j],range_x_diff_local,range_y_diff_local,band_limit,predicted_density[j]),
            }
          else:
            plot_dict = {
              'true_energy': true_density_plot[j],
              'predicted_energy': predicted_density[j],
            }
          if input_pose_density is not None:
            plot_dict['input_pose_density'] = input_pose_density[j]
        # plot_density(ground_truth,measurement,range_x,range_y,folder_path,epoch,dict_density,title,iter=None):
        plot_density(ground_truth[j],measurements[j],range_x,range_y,log_dir,plot_dict,f" validation_epoch_{epoch}_batch_{batch_idx}_sample_{j}")

    dict_log = {'Epoch': epoch + 1,
          'Validation NLL': nll_tot / len(data_loader),
          'Validation KL Div 2': kl_2_tot / len(data_loader),
          'Validation KL Div 1': kl_1_tot / len(data_loader),
          'Validation MAE': mae_tot / len(data_loader),
          'Validation RMSE': rmse_tot / len(data_loader),
          'True NLL': true_nll_tot / len(data_loader)}
    if rmse_cov_tot:
      dict_log['Validation RMSE Cov'] = rmse_cov_tot / len(data_loader)
    
    return dict_log
  
def main(args):
  # Data Parameters
  n_samples = args.n_samples
  trajectory_length = args.trajectory_length
  measurement_noise = args.measurement_noise
  initial_noise = args.initial_noise
  range_x = (args.range_x_start, args.range_x_end)
  range_y = (args.range_y_start, args.range_y_end)
  band_limit = args.band_limit
  scaling_factor = 1
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  num_epochs = args.num_epochs

  true_cov = torch.tensor([measurement_noise**2,measurement_noise**2]).unsqueeze(0)
  initial_cov = torch.tensor([initial_noise**2,initial_noise**2]).unsqueeze(0)

  range_x_diff = range_x[1] - range_x[0]
  range_y_diff = range_y[1] - range_y[0]
  range_x_diff_local = args.range_x_diff_local
  range_y_diff_local = args.range_y_diff_local
  if range_x_diff_local is not None and range_y_diff_local is not None:
    print("Using local grid")
    localgrid = True
    step_t = (round(range_x_diff_local/band_limit[0], 2), round(range_y_diff_local/band_limit[1], 2))
  else:
    print("Using the whole grid")
    localgrid = False
    step_t = (round(range_x_diff/band_limit[0], 2), round(range_y_diff/band_limit[1], 2))

  
  # Scheduler for regularization parameter lambda_
  # Higher the value of reg_factor, faster the decay of lambda_
  lambda_scheduler = lambda epoch: args.start_lambda * torch.exp(torch.tensor(-args.reg_factor * epoch / num_epochs))

  data_path = os.path.join(base_path, 'data')
  os.makedirs(data_path, exist_ok=True)
  data_generator = TrajectoryGenerator(range_x, range_y, step_t, n_samples , trajectory_length, measurement_noise)
  train_loader, val_loader = data_generator.create_data_loaders(data_path, batch_size,flag_flattend=True)
  
  if args.gaussian_param:
    model = SimpleModel()
  else:
    if args.architecture == 0:
      print("Using MLP CNN Model")
      model = DensityPredictorMLPCNN(2, band_limit, batch_size)
      model.apply(initialize_weights)
    elif args.architecture == 1:
      print("Using CNN Model")
      model = DensityPredictorCNN(2, band_limit, batch_size)
      model.apply(initialize_weights)
    elif args.architecture == 2:
      print("Using CNN Model with batch norm")
      model = CNNModel_A(band_limit)
      model.apply(init_weights_identity)
    elif args.architecture == 3:
      print("Using MLP")
      model = NeuralNetwork(2, band_limit,batch_size)
      model.apply(init_weights_mlp)
    else:
      raise ValueError(f"Unsupported architecture type: {args.architecture}")

  if args.optimizer == 'adam':
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  elif args.optimizer == 'rmsprop':
      optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
  elif args.optimizer == 'sgd':
      optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  else:
      raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

  # Creating folders to log
  if localgrid:
    run_name = "UE R2 Local Grid"
  else:
    run_name = "UE R2"
  current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  random_number = random.randint(1000, 9999)
  log_dir = os.path.join(base_path,"logs", run_name, current_datetime + "_" + str(random_number))
  os.makedirs(log_dir, exist_ok=True)
  if localgrid:
    hed = HarmonicExponentialDistribution(band_limit,step_t,range_x_diff=range_x_diff_local,range_y_diff=range_y_diff_local)
  else:
    hed = HarmonicExponentialDistribution(band_limit,step_t,range_x,range_y)
  batch_step = 0
  log_freq = len(train_loader)/3
  for epoch in range(num_epochs):
    loss_tot = 0
    kl_2_tot = 0
    true_nll_tot = 0
    mae_tot = 0
    rmse_tot = 0
    kl_1_tot = 0
    nll_tot = 0
    rmse_cov_tot = 0
    val_dict = validate_model(model, val_loader, hed, args, localgrid, epoch, log_dir,range_x, range_y)
    wandb.log(val_dict)
    if args.decay_lr:
      learning_rate_decay = args.learning_rate_start + (args.learning_rate_end - args.learning_rate_start) * (1 - np.exp((-args.lr_factor * epoch / num_epochs)))
      print("learning_rate",learning_rate_decay)
      for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_decay
    if args.reg_factor:
      lambda_ = lambda_scheduler(epoch)
    print("regularisation factor",lambda_)
    print(f"Training Epoch: {epoch + 1}")
    sample_batch = random.uniform(0, len(train_loader))
    for batch_idx, (ground_truth, measurements) in enumerate(train_loader):
      check_model_weights_nan(model)
      if localgrid:
          input_pose_pdf = GaussianDistribution(ground_truth, initial_cov, band_limit, range_x_diff=range_x_diff_local, range_y_diff=range_y_diff_local)
          input_pose_density = input_pose_pdf.density_over_local_grid()
      else:
        input_pose_pdf = GaussianDistribution(ground_truth,initial_cov,band_limit, range_x,range_y)
        input_pose_density = input_pose_pdf.density_over_grid()
      if args.gaussian_param:
        mu_x,mu_y,logcov_x, logcov_y = model(ground_truth)
        mu = torch.cat((mu_x,mu_y),dim=1)
        cov = torch.cat((torch.exp(logcov_x),torch.exp(logcov_y)),dim=1)
        mu_ = mu.clone().detach()
        cov_ = cov.clone().detach()
        if localgrid:
          predicted_distribution = GaussianDistribution(mu,cov,band_limit,range_x_diff=range_x_diff_local,range_y_diff=range_y_diff_local)
          predicted_density = predicted_distribution.density_over_local_grid()
        else:
          predicted_distribution = GaussianDistribution(mu,cov,band_limit,x_range=range_x,y_range=range_y)
          predicted_density = predicted_distribution.density_over_grid()
      else: 
        if args.architecture == 0:
          outputs = model(ground_truth)
        elif args.architecture == 1:
          outputs = model(input_pose_density)
        elif args.architecture == 2:
          outputs = model(input_pose_density)
        elif args.architecture == 3:
          outputs = model(ground_truth)
          # hed.convolve(input_pose_density,outputs)
        else:
          raise ValueError(f"Unsupported architecture type: {args.architecture}")
        check_tensor_nan(outputs)
        z = hed.normalization_constant(outputs)
        check_tensor_nan(z)
        outputs = outputs/z
        predicted_density = outputs
        # print(predicted_density)
      target_distribution = GaussianDistribution(ground_truth, true_cov,band_limit,x_range=range_x,y_range=range_y)
      true_nll_input = target_distribution.negative_log_likelihood(measurements)
      true_density_plot = target_distribution.density_over_grid()

      if args.gaussian_nll and args.gaussian_param:
        loss = predicted_distribution.negative_log_likelihood(measurements)
        nll = loss
        rmse_tot += root_mean_square_error(mu_,ground_truth).item()
        mae_tot += mean_absolute_error(mu_,ground_truth).item()
        measurement_cov  = torch.ones_like(cov_)* (measurement_noise**2)
        rmse_cov_tot += root_mean_square_error(cov_,measurement_cov).item()
      else:
        if localgrid:
          loss = hed.loss_regularisation_norm(lambda_,predicted_density, measurements,mean=ground_truth,scaling_factor=scaling_factor)
          nll = torch.mean(hed.negative_log_likelihood_local_grid(predicted_density, measurements,mean=ground_truth))
          target_distribution_local_grid = GaussianDistribution(ground_truth,true_cov,band_limit,range_x_diff=range_x_diff_local,range_y_diff=range_y_diff_local)
          true_density_metric = target_distribution_local_grid.density_over_local_grid()
          mode = hed.mode(predicted_density, ground_truth)
        else:
          loss = hed.loss_regularisation_norm(lambda_,predicted_density, measurements,scaling_factor=scaling_factor)
          nll = torch.mean(hed.loss_energy(predicted_density, measurements))
          true_density_metric = true_density_plot
          mode = hed.mode(predicted_density)
        rmse_tot += root_mean_square_error(mode,ground_truth).item()
        mae_tot += mean_absolute_error(mode,ground_truth).item()
      kl_div_2 = kl_divergence(true_density_metric , predicted_density)
      kl_div_1 = kl_divergence(predicted_density, true_density_metric )
      optimizer.zero_grad()
      loss.backward()
      if batch_step % 100 == 0:
        for name, param in model.named_parameters():
          if param.grad is not None:
            wandb.log({'batch': batch_step,f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
      optimizer.step()
      loss_tot +=loss.item()
      kl_2_tot += kl_div_2.item()
      kl_1_tot += kl_div_1.item()
      true_nll_tot += true_nll_input.item()
      nll_tot += nll.item()
      wandb.log({'batch': batch_step, 'Train batch Loss': loss.item(), 'Train batch KL Div 2': kl_div_2.item(), 'Train batch KL Div 1': kl_div_1.item(), 'Train batch NLL': nll.item()})
      batch_step += 1
      if epoch % 50 == 0 and batch_idx == sample_batch:
        indices = np.random.choice(args.batch_size, 5, replace=False)
        for j in indices:
          plot_dict = {}
          if localgrid:
            plot_dict = {
              'true_energy': true_density_plot[j],
              'predicted_energy': fit_grid_into_larger(ground_truth[j],range_x_diff_local,range_y_diff_local,band_limit,predicted_density[j]),
            }
          else:
            plot_dict = {
              'true_energy': true_density_plot[j],
              'predicted_energy': predicted_density[j],
            }
          if input_pose_density is not None:
            plot_dict['input_pose_density'] = input_pose_density[j]
        # plot_density(ground_truth,measurement,range_x,range_y,folder_path,epoch,dict_density,title,iter=None):
        plot_density(ground_truth[j],measurements[j],range_x,range_y,log_dir,plot_dict,f"training_epoch_{epoch}_batch_{batch_idx}_sample{j}")

    dict_log = {'Epoch': epoch + 1,
              'Train loss': loss_tot / len(train_loader),
              'Train NLL': nll_tot / len(train_loader),
              'Train KL Div 2' : kl_2_tot / len(train_loader), 
              'Train KL Div 1' : kl_1_tot / len(train_loader),
              'Train MAE' : mae_tot/len(train_loader),
              'Train RMSE' : rmse_tot/len(train_loader),
              'True NLL' : true_nll_tot/len(train_loader)}
    if rmse_cov_tot:
      dict_log['RMSE Cov'] = rmse_cov_tot/len(train_loader)
    wandb.log(dict_log)
  print("Training Completed")
  for img_file in os.listdir(log_dir):
      if img_file.endswith(".png"):
        wandb.log({img_file: wandb.Image(os.path.join(log_dir, img_file))})
  wandb.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='CNN Uncertainty Estimation in R2')
  parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
  parser.add_argument('--trajectory_length', type=int, default=100, help='Trajectory length')
  parser.add_argument('--measurement_noise', type=float, default=0.15, help='Measurement noise')
  parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
  parser.add_argument('--initial_noise', type=float, default=0.05, help='Initial noise')
  parser.add_argument('--band_limit', type=str, default="50 50", help='Band limit')
  parser.add_argument('--range_x_start', type=float, default=-0.5, help='Range x start')
  parser.add_argument('--range_x_end', type=float, default=0.5, help='Range x end')
  parser.add_argument('--range_y_start', type=float, default=-0.5, help='Range y start')
  parser.add_argument('--range_y_end', type=float, default=0.5, help='Range y end')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (adam, rmsprop, sgd)')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--decay_lr', type=int, default=0, help='Decay learning rate')
  parser.add_argument('--range_x_diff_local', type=str, default="0.5", help='Local range x difference')
  parser.add_argument('--range_y_diff_local', type=str, default="0.5", help='Local range y difference')
  parser.add_argument('--architecture', type=int, choices=[0, 1, 2, 3, 4], default=2, help='Choice of architecture (0, 1, 2, 3, 4)')
  parser.add_argument('--start_lambda', type=float, default=100, help='Starting value of lambda for regularization')
  parser.add_argument('--reg_factor', type=float, default=20, help='Regularization factor for lambda scheduler')
  parser.add_argument('--learning_rate_start', type=float, default=0.001, help='Starting learning rate for decay')
  parser.add_argument('--learning_rate_end', type=float, default=0.0001, help='Ending learning rate for decay')
  parser.add_argument('--lr_factor', type=float, default=10, help='Factor for learning rate adjustment')
  parser.add_argument('--gaussian_param', type=int, default=1, help='Use Gaussian parameterization')
  parser.add_argument('--gaussian_nll', type=int, default=0, help='Use Gaussian NLL')
  args = parser.parse_args()
  band_limit = [int(x) for x in args.band_limit.split()]
  args.band_limit = band_limit
  if (args.range_x_diff_local == "None" or args.range_x_diff_local == None) and (args.range_y_diff_local == "None" or args.range_y_diff_local == None):
      range_x_diff_local = None  # Return as is if the string is "None"
      range_y_diff_local = None
  else:
      range_x_diff_local=float(args.range_x_diff_local)  # Attempt to convert to float
      range_y_diff_local=float(args.range_y_diff_local)
  args.range_x_diff_local = range_x_diff_local
  args.range_y_diff_local = range_y_diff_local
  torch.manual_seed(args.seed)

    # If you are using CUDA
  if torch.cuda.is_available():
      torch.cuda.manual_seed(args.seed)

  wandb.init(project="Diff-HEF",group="R2",entity="korra141",
              tags=["R2", "Uncertainity Estimation", "CNN", "predicting density","predicting delta density","measurement likelihood","unimodal"],
              name="R2-UncertainityEstimation",
              config=args)
  main(args)