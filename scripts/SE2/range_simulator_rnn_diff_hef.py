import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import sys
import math
import datetime
import random
import pdb
import psutil
import wandb
import torch.profiler

base_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(base_path)
pid = os.getpid()

torch.cuda.empty_cache()
from src.distributions.SE2.GaussianDistribution import GaussianSE2 as GaussianDistribution_se2
from src.distributions.R1.HarmonicExponentialDistribution import HarmonicExponentialDistribution as R1_HED
from src.distributions.SE2.SE2_FFT import SE2_FFT
from src.utils.sampler import se2_grid_samples_torch
from src.filter.HEF_SE2 import HEFilter
from src.data_generation.SE2.range_simulator import generate_bounded_se2_dataset
from src.data_generation.SE2.range_simulator import SE2Group
from src.utils.metrics import rmse_se2, compute_weighted_mean
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DensityEstimator(nn.Module):
    def __init__(self, grid_size=(50, 50, 32)):
        super(DensityEstimator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softplus = nn.Softplus()
        self.input_padding = nn.ReplicationPad3d(1)
        self.grid_size = grid_size

    def forward(self, x):
        x = self.input_padding(x)
        x = self.leaky_relu(self.conv1(x))
        # x = self.leaky_relu(self.conv2(x))
        x = self.conv4(x)
        x = self.softplus(x)
        x = x.squeeze(1)
        x = x[:, 1:1+self.grid_size[0], 1:1+self.grid_size[1], 1:1+self.grid_size[2]]
        return x



class CustomRNN(nn.Module):
    def __init__(self, grid_size, cov_prior, MOTION_NOISE, batch_size):
        super(CustomRNN, self).__init__()
        range_x = (-0.5, 0.5)
        range_y = (-0.5, 0.5)
        self.custom_cell = DensityEstimator(grid_size).to(device)
        initialize_weights(self.custom_cell)
        self.optimizer = optim.Adam(self.custom_cell.parameters(), lr=0.0001)
        self.cov_prior = cov_prior
        self.grid_size = grid_size
        self.fft = SE2_FFT(spatial_grid_size=self.grid_size,
                       interpolation_method='spline',
                       spline_order=1,
                       oversampling_factor=1)
        self.hed_r1 = R1_HED(math.prod(self.grid_size), torch.sqrt(torch.tensor(2)))
        self.MOTION_NOISE = MOTION_NOISE
        self.batch_size = batch_size
        self.diff_hef_filter = HEFilter(self.grid_size, range_x, range_y)



    def forward(self, inputs, measurements, control):
        # if hx is None:
        #     hx = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        poses, X, Y, T = se2_grid_samples_torch(self.batch_size, self.grid_size)
        poses, X, Y, T = poses.to(inputs.device), X.to(inputs.device), Y.to(inputs.device), T.to(inputs.device)
        prior_pdf_diff = GaussianDistribution_se2(inputs[:, 0], self.cov_prior, self.grid_size).density(poses).reshape(-1, *self.grid_size)
        output_pose = []
        loss_tot = 0
        nll_posterior_tot = 0
        for i in range(inputs.size(1)):  # Iterate over sequence length
            posterior_pdf_diff, predicted_pose, loss, nll_posterior = diff_hef_step(prior_pdf_diff, inputs[:, i], measurements[:, i], control[:, i], poses, X, Y, T, self.grid_size, self.diff_hef_filter, self.custom_cell , self.fft, self.hed_r1, self.optimizer, self.MOTION_NOISE, self.batch_size)  
            prior_pdf_diff = posterior_pdf_diff
            output_pose.append(predicted_pose)
            loss_tot += loss
            nll_posterior_tot += nll_posterior
        return torch.stack(output_pose, dim=1), loss_tot/inputs.size(1), nll_posterior_tot/inputs.size(1)


def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.01)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.01)

def diff_hef_step(prior_pdf, inputs, measurements, control, poses, X, Y, T, grid_size, hef_filter, model, fft, hed_r1, optimizer, MOTION_NOISE, batch_size):
    with torch.no_grad():
        process = GaussianDistribution_se2(control, torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float32).to(device), grid_size)
        process_pdf = process.density(poses).reshape(-1, *grid_size)
        eta_bel_x_t_bar, density_bel_x_t_bar = hef_filter.predict(prior_pdf, process_pdf)
        density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32).to(device)
    predicted_measurement_density = model(density_bel_x_t_bar.unsqueeze(1))
    start_time_post_prediction = datetime.datetime.now()
    _, z_se2 = fft.compute_moments_lnz(predicted_measurement_density)
    z_se2_processed = torch.where(z_se2 == 0, torch.ones_like(z_se2), z_se2)
    if torch.isnan(z_se2_processed).any() or torch.isnan(predicted_measurement_density).any():
        pdb.set_trace()
    predicted_measurement_density = predicted_measurement_density / z_se2_processed.view(-1, 1, 1, 1)

    predicted_measurement_density_flat = predicted_measurement_density.view(batch_size, -1)
    loss = hed_r1.negative_log_likelihood(predicted_measurement_density_flat, measurements)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        start_time_update = datetime.datetime.now()
        posterior_energy = hef_filter.update(eta_bel_x_t_bar, torch.log(predicted_measurement_density + 1e-8))
        posterior_pdf = torch.exp(posterior_energy).to(inputs.device)
        predicted_pose = compute_weighted_mean(posterior_pdf, poses, X, Y, T)
        start_time_nll = datetime.datetime.now()
        nll_posterior = torch.mean(fft.neg_log_likelihood(posterior_energy, inputs))

    return posterior_pdf, predicted_pose, loss.item(), nll_posterior.item()

def analytic_hef(prior_pdf, inputs, measurements_energy, control, poses, X, Y, T,  grid_size, hef_filter, fft, MOTION_NOISE):
    process = GaussianDistribution_se2(control, torch.diag(torch.tensor(MOTION_NOISE ** 2)).to(torch.float32).to(device), grid_size)
    process_pdf = process.density(poses).reshape(-1, *grid_size)
    eta_bel_x_t_bar, density_bel_x_t_bar = hef_filter.predict(prior_pdf, process_pdf)
    density_bel_x_t_bar = density_bel_x_t_bar.to(torch.float32)

    _, _, _, _, _, eta = fft.analyze(measurements_energy.reshape(-1, *grid_size))
    measurement_energy, _, _, _, _, eta = fft.synthesize(eta)
    measurement_energy = measurement_energy.reshape(-1, *grid_size)
    lnz, z_m = fft.compute_moments_lnz(measurement_energy)
    measurement_energy = measurement_energy - lnz.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    posterior_energy = hef_filter.update(eta_bel_x_t_bar, measurement_energy)
    posterior_energy = posterior_energy.to(torch.float32)
    
    mean_nll = torch.mean(fft.neg_log_likelihood(posterior_energy, inputs))
    posterior_pdf = torch.exp(posterior_energy).to(inputs.device)
    predicted_pose = compute_weighted_mean(posterior_pdf, poses, X, Y, T)

    return posterior_pdf, predicted_pose , mean_nll

def main(args):
    # logging file 
    run_name = "SE2_Range_DiffHEF"
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    # Shared run_id across all processes
    logging_path = os.path.join(base_path, "logs", run_name, current_datetime + "_" + str(random_number))
    os.makedirs(logging_path, exist_ok=True)
    # parameters
    NUM_EPOCHS = args.num_epochs
    NUM_TRAJECTORIES = args.num_trajectories
    TRAJECTORY_LENGTH = args.trajectory_length
    STEP_MOTION = SE2Group(args.step_motion[0], args.step_motion[1], args.step_motion[2])
    MOTION_NOISE = np.sqrt(np.array(args.motion_cov))
    MEASUREMENT_NOISE = np.sqrt(args.measurement_cov)
    batch_size = args.batch_size
    validation_split = args.validation_split
    grid_size = tuple(args.grid_size)
    cov_prior = torch.diag(torch.tensor(args.cov_prior)).to(device)
    
    poses, X, Y, T = se2_grid_samples_torch(batch_size, grid_size)
    poses, X, Y, T = poses.to(device), X.to(device), Y.to(device), T.to(device)
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # Generate dataset
    train_loader = generate_bounded_se2_dataset(
        num_trajectories=NUM_TRAJECTORIES,
        trajectory_length=TRAJECTORY_LENGTH,
        step_motion=STEP_MOTION,
        motion_noise=MOTION_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
        samples=poses.cpu().numpy(),
        batch_size=batch_size
    )

    range_x = (-0.5, 0.5)
    range_y = (-0.5, 0.5)
    # Initialize the model
    rnn_ = CustomRNN(grid_size, cov_prior, MOTION_NOISE, batch_size)
    rnn_.to(device)
    rnn_.train()

    # model = DensityEstimator(grid_size).to(device)
    # initialize_weights(model)
    # Initialize the HEF filter
    # diff_hef_filter = HEFilter(grid_size, range_x, range_y)
    true_hef_filter = HEFilter(grid_size, range_x, range_y)
    # hed_r1 = R1_HED(math.prod(grid_size), torch.sqrt(torch.tensor(2)))
    # Initialize the SE2 FFT
    # diff_fft = SE2_FFT(spatial_grid_size=grid_size,
    #                    interpolation_method='spline',
    #                    spline_order=1,
    #                    oversampling_factor=1)
    true_fft = SE2_FFT(spatial_grid_size=grid_size,
                       interpolation_method='spline',
                       spline_order=1,
                       oversampling_factor=1)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)


    # posterior_pdf_diff, predicted_pose, loss, nll_posterior = diff_hef_step(prior_pdf_diff, inputs[:, i], measurements[:, i], control[:, i], poses, X, Y, T, grid_size, diff_hef_filter, model, diff_fft, hed_r1, optimizer, MOTION_NOISE, batch_size)
    with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
      on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
      record_shapes=True,
      profile_memory=True
  ) as prof:           
        for epoch in range(NUM_EPOCHS):
            loss_tot = 0
            mean_rmse_tot = 0
            mean_rmse_true_tot = 0
            nll_posterior_tot = 0
            nll_posterior_true_tot = 0
            start_time = datetime.datetime.now()
            for j, data in enumerate(train_loader):
                start_time_step = datetime.datetime.now()
                inputs, measurements, measurements_energy, control = data
                inputs, measurements, measurements_energy, control = inputs.to(device), measurements.to(device), measurements_energy.to(device), control.to(device)
                prior_pdf = GaussianDistribution_se2(inputs[:, 0], cov_prior, grid_size).density(poses).reshape(-1, *grid_size)
                prior_pdf = prior_pdf.to(device)
                prior_pdf_true = prior_pdf
                # prior_pdf_diff = prior_pdf
                # trajectory_list = []
                trajectory_list_true = []
                predicted_trajectory, measurement_nll, posterior_nll = rnn_(inputs, measurements, control)
                process = psutil.Process(pid)
                mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                print(f"Memory Usage model training | after:{mem:.2f} GB")
                loss_tot += float(measurement_nll)
                nll_posterior_tot += float(posterior_nll)
                
                # Perform operations on inputs and labels using HEF analytical filter here
                for i in range(TRAJECTORY_LENGTH):
                    
                    # process = psutil.Process(pid)
                    # mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                    # posterior_pdf_diff, predicted_pose, loss, nll_posterior = diff_hef_step(prior_pdf_diff, inputs[:, i], measurements[:, i], control[:, i], poses, X, Y, T, grid_size, diff_hef_filter, model, diff_fft, hed_r1, optimizer, MOTION_NOISE, batch_size)
                    posterior_pdf_true, predicted_pose_true , nll_posterior_true = analytic_hef(prior_pdf_true, inputs[:,i], measurements_energy[:,i], control[:,i], poses, X, Y, T, grid_size, true_hef_filter, true_fft, MOTION_NOISE)
                    # process = psutil.Process(pid)
                    # mem_1 = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                    # print(f"Memory Usage model training | before:{mem:.2f} GB, after {mem_1:.2f} GB")
                    prior_pdf_true = posterior_pdf_true
                    # with torch.no_grad():
                        # trajectory_list.append(predicted_pose)
                    trajectory_list_true.append(predicted_pose_true)
                    # nll_posterior_tot += float(nll_posterior.item())
                    nll_posterior_true_tot += float(nll_posterior_true.item())
                    
                print(f"Time taken for step {j}: {datetime.datetime.now() - start_time_step}")
                
                # predicted_trajectory = torch.stack(trajectory_list, dim=1)
                predicted_trajectory_true = torch.stack(trajectory_list_true, dim=1)
                # del posterior_pdf_diff, predicted_pose, loss, nll_posterior
                del posterior_pdf_true, predicted_pose_true, nll_posterior_true
                torch.cuda.empty_cache()
                with torch.no_grad():
                    mean_rmse_tot += rmse_se2(inputs, predicted_trajectory)
                    mean_rmse_true_tot += rmse_se2(inputs, predicted_trajectory_true)
                prof.step()

            # Create a table for logging
            end_time = datetime.datetime.now()
            table_data = [
                ["Epoch", epoch],
                ["Time", str(end_time - start_time)],
                ["Loss", loss_tot / len(train_loader)],
                ["Diff HEF NLL Posterior", nll_posterior_tot / len(train_loader)],
                ["True HEF NLL Posterior", nll_posterior_true_tot / (TRAJECTORY_LENGTH * len(train_loader))],
                ["Diff HEF RMSE", mean_rmse_tot / len(train_loader)],
                ["True HEF RMSE", mean_rmse_true_tot / len(train_loader)]
            ]
            # wandb.log({ "Epoch": epoch, "Time": str(end_time - start_time), "Loss": torch.mean(measurement_nll).item(), "Diff HEF NLL Posterior": torch.mean(posterior_nll).item(), "True HEF NLL Posterior": nll_posterior_true_tot / (TRAJECTORY_LENGTH * len(train_loader)), "Diff HEF RMSE": mean_rmse_tot / len(train_loader), "True HEF RMSE": mean_rmse_true_tot / len(train_loader) })

            # Print the table
            print("\n" + "-" * 40)
            for row in table_data:
                print(f"{row[0]:<30} {row[1]:<10}")
            print("-" * 40)
        # Save the model
    model_save_path = os.path.join(logging_path, "measurement_model.pth")
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path, base_path=loggin_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Diff-HEF SE2 Range Simulator")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num_trajectories', type=int, default=500, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=20, help='Length of each trajectory')
    parser.add_argument('--step_motion', type=float, nargs=3, default=[0.01, 0.00, np.pi / 40], help='Step motion parameters')
    parser.add_argument('--motion_cov', type=float, nargs=3, default=[0.1, 0.1, 0.04], help='Motion noise parameters')
    parser.add_argument('--measurement_cov', type=float, default=0.5, help='Measurement noise')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--grid_size', type=int, nargs=3, default=[16, 16, 16], help='Grid size')
    parser.add_argument('--cov_prior', type=float, nargs=3, default=[0.01, 0.01, 0.01], help='Covariance prior')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # run = wandb.init(mode="disabled", project="Diff-HEF",group="SE2",entity="korra141",
    #           tags=["SE2","DenistyEstimation","UnimodalNoise","IndependentSpace", "Trial"],
    #           name="SE2-DiffHEF-RangeSimulator-1",
    #           notes="Diff-HEF on SE2 Range Simulator",
    #           config=args)
    # artifact = wandb.Artifact("SE2_Range_DiffHEF", type="script")
    # artifact.add_file(__file__)
    # run.log_artifact(artifact)
    main(args)
