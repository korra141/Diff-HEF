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
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution
from src.utils.visualisation import plot_circular_distribution, plotting_von_mises
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
    parser.add_argument('--measurement_noise_min', type=float, default=0.1, help='Standard deviation of the measurement noise')
    parser.add_argument('--measurement_noise_max', type=float, default=0.3, help='Standard deviation of the measurement noise')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step between poses in trajectory')
    parser.add_argument('--shuffle_flag', type=bool, default=True, help='Whether to shuffle the samples')
    parser.add_argument("--log-freq", type=int, default=10, help="Frequency of logging the results")
    parser.add_argument('--range_theta', type=float, default=np.pi/2, help='Range of theta for local grid')
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


def main(args):

    # Generate training data
    data_path =  os.path.join(base_path, 'data')
    train_loader =  generating_data_S1_heteroscedastic(data_path, args, args.batch_size, args.n_samples, args.trajectory_length, args.measurement_noise_min, args.step_size, True)
    # test_loader = generating_data_S1(args, args.batch_size, 20, args.trajectory_length, args.measurement_noise_min, args.step_size, False)

    # Initialize the model, optimizer, and loss function
    model = EnergyNetwork(1, args.hidden_size, args.band_limit)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    if args.range_theta == None:
        run_name = "S1-UE-Heteroscedastic"    
    else:
        run_name = "S1-UE-Heteroscedastic-Local"

    logging_path = os.path.join(base_path,"logs", run_name, str(ctime))
    os.makedirs(logging_path)
    hed = HarmonicExponentialDistribution(args.band_limit, args.range_theta)
    for epoch in range(args.num_epochs):
        loss_tot = 0
        kl_div_tot = 0
        for i, (ground_truth, measurements) in enumerate(train_loader):
          energy = model(ground_truth)
          if args.range_theta == None:
            loss = hed.negative_log_likelihood(energy, measurements)
          else:
            loss = hed.negative_log_likelihood_local(energy, measurements, ground_truth)
          true_distribution = VonMissesDistribution(ground_truth, heteroscedastic_noise(ground_truth, args.measurement_noise_min, args.measurement_noise_max),args.band_limit)
          if args.range_theta == None:
            true_density = true_distribution.density()
          else:
            true_density = true_distribution.density_local(args.range_theta)
          predicted_density = torch.exp(energy - hed.normalization_constant(energy))
          kl_div_tot += kl_divergence_s1(true_density, predicted_density)
          if i == 0 and epoch % args.log_freq == 0:
            fig, ax = plt.subplots()
            if args.range_theta == None:
              ax = plot_circular_distribution(energy[0],legend="predicted distribution",ax=ax)
            else:
              ax = plot_circular_distribution(energy[0],legend="predicted distribution",mean=ground_truth[0,0],range_theta=args.range_theta,ax=ax)
            ax = plotting_von_mises(ground_truth[0],heteroscedastic_noise(ground_truth[0], args.measurement_noise_min, args.measurement_noise_max).item()**2, args.band_limit,ax,"true distribution")
            ax.plot(torch.cos(measurements[0]),torch.sin(measurements[0]),'o',label="measurement data")
            ax.plot(torch.cos(ground_truth[0]), torch.sin(ground_truth[0]), 'o', label="pose data")
            ax.set_title(f"Epoch {epoch,}", loc='center')
            ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
            plt.savefig(os.path.join(logging_path, f"epoch_{epoch}.png"), format='png', dpi=300)
            # plt.show()
            plt.close()

            # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_tot += loss.item()
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_tot/len(train_loader):.4f}')
        wandb.log({
          'Epoch': epoch + 1,
          'NLL Loss': loss_tot / len(train_loader),
          'KL Divergence': kl_div_tot / len(train_loader)})
    # Log all the generated images to wandb
    for img_file in os.listdir(logging_path):
        if img_file.endswith(".png"):
            wandb.log({f"{img_file}": wandb.Image(os.path.join(logging_path, img_file))})
    print("Training finished!")


if __name__ == '__main__':
   args = parse_args()
   wandb.init(project="Diff-HEF",group="S1",entity="korra141",
            tags=["S1","UncertainityEstimation","HeteroscedasticNoise"],
            name="S1-UncertainityEstimation",
            config=args)
   main(args)