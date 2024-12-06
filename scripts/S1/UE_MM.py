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

from src.data_generation.S1.toy_dataset import generating_data_S1_multimodal
from src.distributions.S1.HarmonicExponentialDistribution import HarmonicExponentialDistribution
from src.distributions.S1.WrappedNormalDitribution import MultimodalGaussianDistribution
from src.utils.visualisation import plot_circular_distribution
from src.utils.metrics import kl_divergence_s1

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment to learn the measurement distribution in S1 using Harmonic Exponential Functions.')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=10, help='Number of neurons in the hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--band_limit', type=int, default=100, help='Band limit')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--trajectory_length', type=int, default=100, help='Length of each trajectory')
    parser.add_argument('--measurement_noise', type=float, default=0.1, help='Standard deviation of the measurement noise')
    # parser.add_argument('--measurement_noise_max', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument('--shuffle_flag', type=bool, default=True, help='Whether to shuffle the samples')
    parser.add_argument("--log-freq", type=int, default=10, help="Frequency of logging the results")
    parser.add_argument("--mean_offset", type=float, default=np.pi/2, help="Mean offset for the multimodal noise")
    parser.add_argument("--n_modes", type=int, default=2, help="Number of modes for the multimodal noise")
    parser.add_argument('--range_theta', type=float, default=np.pi, help='Range of theta for local grid')
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


def main(args):
    
    measurement_noise = torch.ones(args.n_modes) * args.measurement_noise
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_multimodal(measurement_noise, args.mean_offset,args.n_modes, data_path, args.batch_size, args.n_samples, args.trajectory_length,  args.step_size, True)
    model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    centers = torch.tile(torch.linspace(-args.mean_offset / 2, args.mean_offset / 2, args.n_modes)[None,:,None], (args.batch_size, 1,1)) # n_modes
    
    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    if args.range_theta == None:
        run_name = "S1-UE-MM"    
    else:
        run_name = "S1-UE-MM-Local"
    logging_path = os.path.join(base_path,"logs", run_name, str(ctime))
    os.makedirs(logging_path)
    # Loss function
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)

    for epoch in range(args.num_epochs):
        loss_tot = 0
        kl_div_tot = 0
        for i, (ground_truth, measurements) in enumerate(train_loader):
            start_time = time.time()
            # Forward pass
            energy = model(ground_truth)

            if args.range_theta == None:
              loss = hed.negative_log_likelihood(energy, measurements)
            else:
              loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
      
            mm_mean = positive_angle(ground_truth.unsqueeze(1) + centers) #(batch_size, n_modes, 2)
            true_distribution = MultimodalGaussianDistribution(mm_mean,measurement_noise,args.n_modes,args.band_limit)
            true_distribution_energy = true_distribution.energy()
            if args.range_theta == None:
              true_distribution_density = true_distribution.density()
            else:
              true_distribution_density = true_distribution.density_local(args.range_theta)
            
            predicted_density = torch.exp(energy - hed.normalization_constant(energy))
            kl_div_tot += kl_divergence_s1(true_distribution_density, predicted_density)

            if i  == 0 and epoch % args.log_freq == 0:
              fig, ax = plt.subplots()
              if args.range_theta == None:
                ax = plot_circular_distribution(energy[0],legend="predicted distribution",ax=ax)
              else:
                ax = plot_circular_distribution(energy[0],legend="predicted distribution",ax=ax,mean=ground_truth[0,0],range_theta=args.range_theta)
              ax = plot_circular_distribution(true_distribution_energy[0],legend="true distribution",ax=ax)
              ax.plot(torch.cos(measurements[0]),torch.sin(measurements[0]),'o',label="measurement data")
              ax.plot(torch.cos(ground_truth[0]), torch.sin(ground_truth[0]), 'o', label="pose data")
              ax.set_title(f"Epoch {epoch,}", loc='center')
              ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
              image_path = os.path.join(logging_path, f"{epoch}.png")
              plt.savefig(image_path, format='png', dpi=300)
              plt.close()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tot += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {loss_tot / len(train_loader)}, KL divergence: {kl_div_tot / len(train_loader)}")
        wandb.log({
              'Epoch': epoch + 1,
              'Train NLL loss': loss_tot / len(train_loader),
              'Train KL divergence': kl_div_tot / len(train_loader),
        })
    for img_file in os.listdir(logging_path):
        if img_file.endswith(".png"):
            wandb.log({f"{img_file}": wandb.Image(os.path.join(logging_path, img_file))})
    print("Training finished!")


if __name__ == '__main__':
   args = parse_args()
   wandb.init(project="Diff-HEF",group="S1",entity="korra141",
              tags=["S1","UncertainityEstimation","MultimodalNoise"],
              name="S1-UncertainityEstimation",
              config=args)
   main(args)
