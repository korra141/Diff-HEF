import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from time import strftime, localtime
import random
import matplotlib.pyplot as plt
import math
import pdb


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.utils.metrics import absolute_error_s1
from src.distributions.S1.WrappedNormalDitribution import VonMissesDistribution,VonMissesDistribution_torch
from src.data_generation.S1.toy_dataset import generating_data_S1_multimodal
def generating_data_S1(batch_size, n_samples, trajectory_length, measurement_noise, recreate=False,step=0.1, shuffle_flag=True):
    """Generates training data for a system with circular motion.

    Args:
      batch_size: The number of samples in each batch.
      n_samples: The total number of samples to generate.
      trajectory_length: The length of each trajectory.
      measurement_noise: The standard deviation of the measurement noise.
      step: The step between poses in trajectory
      shuffle_flag: Whether to shuffle the samples,

    Returns:
      Flattened pose and noisy measurement data in a TensorDataset.
    """
    data_path = os.path.join(base_path, 'data', f's1_simple_dataset_lstm_{measurement_noise}.pt')
    if recreate or not os.path.exists(data_path):
        print('Generating Data and Saving')
        starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        true_trajectories = np.ndarray((n_samples, trajectory_length))
        measurements = np.ndarray((n_samples, trajectory_length))

        for i in range(n_samples):
            # Generate a circular trajectory with a random starting position.
            initial_angle = starting_positions[i]
            trajectory = (initial_angle + np.arange(trajectory_length) * step) % (2 * np.pi)
            true_trajectories[i] = trajectory
            # Add Gaussian noise to the measurements.
            measurements[i] = (trajectory + np.random.normal(0, measurement_noise, trajectory_length) + np.ones_like(trajectory)*(2*math.pi)) % (2 * np.pi)

        measurements_ = torch.from_numpy(measurements)[:, :, None].type(torch.FloatTensor)
        ground_truth_ = torch.from_numpy(true_trajectories)[:, :, None].type(torch.FloatTensor)
        # ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        # measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_, measurements_)
        torch.save(train_dataset, data_path)
    else:
        print('Loading Data')
        train_dataset = torch.load(data_path)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    
    return train_loader, val_loader
    


class LSTMFilter(nn.Module):
    def __init__(self):
        super(LSTMFilter, self).__init__()
        self.input_dim = 2
        self.state_dim = 1
        self.hidden_dim = 4

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # Fully connected layer to map hidden state to estimated state
        # self.fc = nn.Linear(self.hidden_dim, self.state_dim)
        self.state_fc = nn.Linear(self.hidden_dim, self.state_dim)  # Predict state
        self.log_var_fc = nn.Linear(self.hidden_dim, self.state_dim)  # Predict log-variance

    def forward(self, control_inputs, measurements, initial_state):
        """
        Args:
            control_inputs (torch.Tensor): Control inputs of shape (batch_size, T, control_dim).
            measurements (torch.Tensor): Measurements of shape (batch_size, T, measurement_dim).
            initial_state (torch.Tensor): Initial state of shape (batch_size, state_dim).

        Returns:
            estimated_states (torch.Tensor): Estimated states of shape (batch_size, T, state_dim).
        """
        # Combine control inputs and measurements
        inputs = torch.cat([control_inputs, measurements], dim=-1)  # Shape: (batch_size, T, input_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)  # Shape: (batch_size, T, hidden_dim)

        # Map LSTM output to state space
        estimated_states = self.state_fc(lstm_out) % (2*math.pi) # Shape: (batch_size, T, state_dim)

        # Incorporate the initial state (optional)
        estimated_states[:, 0, :] += initial_state  # Add initial state to the first time step

        # Predict log-variance
        log_var = self.log_var_fc(lstm_out)

        return estimated_states, log_var
    
def plotting_function(logging_path, ground_truth, measurements, estimated_states, cov, sample_batch):
    """
    Args:
        loggin_path (str): Path to save the plots
        ground_truth (torch.Tensor): Ground truth states of shape (batch_size, T, state_dim).
        measurements (torch.Tensor): Measurements of shape (batch_size, T, measurement_dim).
        estimated_states (torch.Tensor): Estimated states of shape (batch_size, T, state_dim).
        sample_batch (int): Index of the batch to plot.
    """
    # Plot the ground truth, measurements, and estimated states
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(ground_truth[sample_batch, :, 0], label='Ground Truth')
    ax.plot(measurements[sample_batch, :, 0], label='Measurements')
    ax.plot(estimated_states[sample_batch, :, 0], label='Estimated States')
    time_steps = np.arange(ground_truth.shape[1])
    if cov is not None:
        noise = torch.sqrt(cov[sample_batch, :, 0])
        lower_bound = estimated_states[sample_batch, :, 0] - noise
        upper_bound = estimated_states[sample_batch, :, 0] + noise
        ax.fill_between(time_steps, lower_bound, upper_bound, color='green', alpha=0.2)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title('Ground Truth, Measurements, and Estimated States')
    ax.legend()
    plt.savefig(os.path.join(logging_path, 'plot.png'))
    plt.close()

def loss_fn(predicted_label, predicted_covariance, true_label):
    """
    Args:
        predicted_label (torch.Tensor): Predicted labels.
        true_label (torch.Tensor): True labels.

    Returns:
        loss (torch.Tensor): Loss value.
    """
    # print(predicted_label.shape)
    diff = absolute_error_s1(predicted_label,true_label)
    nll = 0.5 * (diff ** 2 / predicted_covariance) + 0.5 * torch.log(2* math.pi* predicted_covariance)
    return torch.mean(nll)


# Example usage
if __name__ == "__main__":
    # Define dimensions
    control_dim = 1
    measurement_dim = 1
    batch_size = 100
    step_size = 0.1
    traj_len = 100
    T = traj_len  # Sequence length
    n_trajs = 500
    lr = 1e-3
    epochs = 500
    noise_p = 0.2
    model_path = None
    mean_offset = 0.785
    measurement_noise = 0.3
    n_modes = 2
    # measurement_noise = torch.ones(n_modes) * measurement_noise
    data_path = os.path.join(base_path, 'data')


    # model_path = "/network/scratch/r/ria.arora/Diff-HEF/logs/LSTM_S1/2024-12-21 21:07:01_9678/model.pth"

    # Create the LSTM filter
    lstm_filter = LSTMFilter()
    # train_loader = generating_data_S1_multimodal(measurement_noise, mean_offset,n_modes,data_path, batch_size, n_trajs, traj_len, step=0.1, shuffle_flag=True, flattend=False)
    train_loader,test_loader = generating_data_S1(batch_size, n_trajs, traj_len, measurement_noise, True, step_size, True)

    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_filter.parameters(), lr=lr)

    ctime = time.time()
    ctime = strftime('%Y-%m-%d %H:%M:%S', localtime(ctime))
    random_number = random.randint(1000, 9999)
    run_name = "LSTM_S1"
    logging_path = os.path.join(base_path,"logs", run_name, str(ctime) + "_" + str(random_number))
    print(f"Logging path: {logging_path}")
    os.makedirs(logging_path)
    
    if model_path is None:
        for epoch in range(epochs):
            loss_tot = 0
            for i, (ground_truth, measurements) in enumerate(train_loader):
                ground_truth = ground_truth.type(torch.FloatTensor)
                measurements = measurements.type(torch.FloatTensor)
                # print(ground_truth.shape)
                # print(measurements.shape)
                control = torch.ones((batch_size, traj_len, 1)) * step_size 
                # + torch.normal(0, noise_p, size=(batch_size, traj_len, 1))  # Noisy Control
                optimizer.zero_grad()
                # print(ground_truth.shape)
                initial_state = ground_truth[:, 0, :]  # Initial state
                mu, logcov = lstm_filter(control, measurements, initial_state)
                cov = torch.exp(logcov)
                loss = loss_fn(mu, cov, ground_truth)
                loss.backward()
                optimizer.step()
                loss_tot += loss.item()
            print(f"Epoch {epoch+1}, Loss: {loss_tot/len(train_loader)}")
        print('Finished Training')
        model_save_path = os.path.join(logging_path, 'model.pth')
        torch.save(lstm_filter.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        print(f"Loading model from {model_path}")
        lstm_filter.load_state_dict(torch.load(model_path))



    # Test the model
    # with torch.no_grad():
    #     loss_tot = 0
    #     sample_batch = np.random.choice(len(test_loader),1).item()
    #     for i, (ground_truth, measurements) in enumerate(test_loader):
    #         control = torch.ones((batch_size, 1, 1)) * step_size + torch.normal(0, noise_p, size=(batch_size, traj_len, 1))  # Noisy Control
    #         initial_state = ground_truth[:, 0, :]  # Initial state
    #         estimated_states, logcov = lstm_filter(control, measurements, initial_state)
    #         cov = torch.exp(logcov)
    #         loss = loss_fn(estimated_states,cov, ground_truth)
    #         loss_tot += loss.item()
    #         if i==0:
    #             plotting_function(logging_path, ground_truth, measurements, estimated_states, cov, sample_batch)
  
    #     print(f"Test Loss: {loss_tot/len(test_loader)}")
        