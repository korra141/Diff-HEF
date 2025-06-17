import torch
import torch.nn as nn
import numpy as np
import os
import sys
import math
import random
import datetime
import matplotlib.pyplot as plt
import argparse


base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)

from src.utils.sampler import se2_grid_samples_torch
from src.utils.metrics import rmse_se2, mse_trajectory
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset, SE2Group
from src.utils.sampler import se2_grid_samples_torch
from src.utils.visualisation import plot_se2_mean_filters
import pickle

class LSTMFilterSE2(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, state_dim=3):
        super(LSTMFilterSE2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # Fully connected layers to map hidden state to state and covariance
        self.state_fc = nn.Linear(self.hidden_dim, self.state_dim)
        self.log_var_fc = nn.Linear(self.hidden_dim, self.state_dim)

    def forward(self, control_inputs, measurements, initial_state, cov_prior):
        inputs = torch.cat([control_inputs, measurements], dim=-1)  # Combine control and measurements
        lstm_out, _ = self.lstm(inputs)  # Pass through LSTM
        estimated_states = self.state_fc(lstm_out)  # Predict states
        log_var = self.log_var_fc(lstm_out)  # Predict log-variance
        state_result = torch.cat((initial_state.unsqueeze(1), estimated_states), dim=1)
        state_result[..., 2] = (state_result[..., 2] + math.pi) % (2 * math.pi) - math.pi  # Wrap angle
        cov = torch.exp(log_var)
        cov_result = torch.cat((cov_prior.unsqueeze(1), cov), dim=1)
        return state_result, cov_result

def loss_fn(predicted_state, cov,  true_state):
    diff = predicted_state - true_state #[batch_size, trajectory_length, state_dim]
    sigma = torch.sqrt(cov) #[batch_size, trajectory_length, state_dim]
    # cov = torch.diag(cov) #[batch_size, trajectory_length, state_dim]
    # diff[..., 2] = (diff[..., 2] + math.pi) % (2 * math.pi) - math.pi  # Wrap angle
    # precision_matrix = torch.inverse(cov)
    # log_det = torch.logdet(cov)
    # diff_expanded = diff.unsqueeze(1)
    # mahalanobis_dist = torch.bmm(torch.bmm(diff_expanded, precision_matrix),
    # diff.unsqueeze(-1)).squeeze()
    # log_constant = k * torch.log(torch.tensor(2 * torch.pi)) + log_det 
    # log_likelihood = -0.5 * (log_constant + mahalanobis_dist)
    # return torch.mean(-log_likelihood)
    # improve this to multivariate.
    # nll = 0.5 * (diff ** 2 / predicted_covariance) + 0.5 * torch.log(2 * math.pi * predicted_covariance)

    nll = 0.5 * torch.sum((diff ** 2) / (sigma ** 2) + torch.log(sigma ** 2) + torch.log(torch.tensor(2) * torch.pi), dim=2)
    return 0.5* (torch.mean(nll) + mse_trajectory(predicted_state,true_state))
    # return torch.mean(nll)

def train_lstm_se2(args, logging_path):
    # Generate dataset
    device = torch.device(args.device)
    poses, X, Y, T = se2_grid_samples_torch(args.batch_size, args.grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    
    train_loader, val_loader, test_loader = generate_bounded_se2_dataset(
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        step_motion=SE2Group(*args.step_motion),
        motion_noise=np.sqrt(np.array(args.motion_cov)),
        measurement_noise=np.sqrt(args.measurement_cov),
        samples=poses.cpu().numpy(),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split
    )
    cov_prior_batch = torch.tile(torch.tensor(args.cov_prior).unsqueeze(0),[args.batch_size, 1]).to(device)
    # Initialize LSTM filter
    lstm_filter = LSTMFilterSE2(input_dim=4, hidden_dim=64, state_dim=3).to(args.device)
    optimizer = torch.optim.Adam(lstm_filter.parameters(), lr=args.lr)
    
    # Training loop
    loss_list = []
    for epoch in range(args.num_epochs):
        lstm_filter.train()
        total_loss = 0
        for inputs, measurements, control, _ in train_loader:
            inputs, measurements, control = inputs.to(args.device), measurements.to(args.device), control.to(args.device)
            initial_state = inputs[:, 0, :]  # Initial state
            
            predicted_states, cov = lstm_filter(control.to(torch.float32), measurements.to(torch.float32), initial_state, cov_prior_batch)
            loss = loss_fn(predicted_states, cov, inputs).to(torch.float32)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_loader)}")
        loss_list.append(total_loss / len(train_loader))

        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            model_save_path = os.path.join(logging_path, f"lstm_se2_epoch_{epoch + 1}.pth")
            torch.save(lstm_filter.state_dict(), model_save_path)


    plt.figure()
    plt.plot(np.arange(0, args.num_epochs), loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    # Save the loss list as a pickle file
    with open(os.path.join(logging_path, "loss_list.pkl"), "wb") as f:
        pickle.dump(loss_list, f)
    plt.savefig(os.path.join(logging_path, "loss_curve.png"))
    plt.close()

def test_lstm_se2(args, logging_path, model_path):
    # Load model
    device = torch.device(args.device)
    lstm_filter = LSTMFilterSE2(input_dim=4, hidden_dim=64, state_dim=3).to(args.device)
    lstm_filter.load_state_dict(torch.load(model_path))
    lstm_filter.eval()

    poses, X, Y, T = se2_grid_samples_torch(args.batch_size, args.grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    

    # Generate test dataset
    _, _, test_loader = generate_bounded_se2_dataset(
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        step_motion=SE2Group(*args.step_motion),
        motion_noise=np.sqrt(np.array(args.motion_cov)),
        measurement_noise=np.sqrt(args.measurement_cov),
        samples=poses.cpu().numpy(),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split
    )
    cov_prior_batch = torch.tile(torch.tensor(args.cov_prior).unsqueeze(0),[args.batch_size, 1]).to(device)

    total_rmse = 0
    nll_lstm = 0
    for inputs, measurements, control, _ in test_loader:
        inputs, measurements, control = inputs.to(args.device), measurements.to(args.device), control.to(args.device)
        initial_state = inputs[:, 0, :]
        with torch.no_grad():
            predicted_states, cov = lstm_filter(control.to(torch.float32), measurements.to(torch.float32), initial_state, cov_prior_batch)
            nll_lstm +=loss_fn(predicted_states, cov, inputs)
            total_rmse += rmse_se2(inputs, predicted_states)

    print(f"Test RMSE: {total_rmse / len(test_loader)}")
    print(f"Test NLL: {nll_lstm / len(test_loader)}")

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM for SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_trajectories', type=int, default=300, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=80, help='Length of each trajectory')
    parser.add_argument('--step_motion', type=float, nargs=3, default=[0.01, 0.0, np.pi / 40], help='Step motion parameters')
    parser.add_argument('--motion_cov', type=float, nargs=3, default=[0.001, 0.001, 0.001], help='Motion noise parameters')
    parser.add_argument('--measurement_cov', type=float, default=0.0001, help='Measurement noise')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.12, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=50, help='Save interval for model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--cov_prior', type=float, nargs=3, default=[0.1, 0.1, 0.1], help='Prior covariance for state estimation')
    parser.add_argument('--grid_size',  type=float, nargs=3, default=[50, 50, 32],  help='Grid size for SE2 sampling')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_path = os.path.join(base_path, "logs", "LSTM_SE2", current_datetime)
    os.makedirs(logging_path, exist_ok=True)

    train_lstm_se2(args, logging_path)
    model_path = os.path.join(logging_path, "lstm_se2_epoch_100.pth")
    # model_path = "/home/mila/r/ria.arora/scratch/Diff-HEF/logs/LSTM_SE2/20250416_235558/lstm_se2_epoch_100.pth"
    test_lstm_se2(args, logging_path, model_path)
