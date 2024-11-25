import numpy as np
import torch
import os
import pdb

class TrajectoryGenerator:
    def __init__(self, range_x, range_y, step, num_trajectories, trajectory_length, measurement_noise, mean_offset=None, n_modes=None):
        self.range_x = range_x
        self.range_y = range_y
        self.step = step
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.measurement_noise = measurement_noise
        self.mean_offset = mean_offset
        self.n_modes = n_modes

    def sample_mm(self, mean, std, n_samples):
        weights = np.ones(self.n_modes) / self.n_modes
        selected_mode = np.random.choice(range(self.n_modes), p=weights)
        noise_point = np.random.normal(loc=mean[selected_mode], scale=std[selected_mode], size=(n_samples))
        # print(noise_point)
        return noise_point

    def generate_multimodal_trajectories(self):
        x_values = np.arange(self.range_x[0], self.range_x[1], self.step[0])
        y_values = np.arange(self.range_y[0], self.range_y[1], self.step[1])

        grid_points = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

        trajectories = []
        measurements = []

        for _ in range(self.num_trajectories):
            current_point = grid_points[np.random.choice(len(grid_points))]
            trajectory = []
            measurement = []

            for _ in range(self.trajectory_length):
                moves = [
                    [self.step[0], 0],  # Right
                    [-self.step[0], 0], # Left
                    [0, self.step[1]],  # Up
                    [0, -self.step[1]],  # Down
                    [self.step[0], self.step[1]],  # Diagonal Right-Up
                    [-self.step[0], self.step[1]], # Diagonal Left-Up
                    [self.step[0], -self.step[1]], # Diagonal Right-Down
                    [-self.step[0], -self.step[1]] # Diagonal Left-Down
                ]

                valid_move_found = False
                while not valid_move_found:
                    move = moves[np.random.choice(len(moves))]
                    next_point = current_point + move
                    centers = np.linspace(-self.mean_offset / 2, self.mean_offset / 2, self.n_modes)
                    
                    noisy_point = ((next_point + self.sample_mm(centers, self.measurement_noise, 2)) // self.step) * self.step

                    if self.range_x[0] <= next_point[0] < self.range_x[1] and self.range_y[0] <= next_point[1] < self.range_y[1]:
                        if self.range_x[0] <= noisy_point[0] < self.range_x[1] and self.range_y[0] <= noisy_point[1] < self.range_y[1]:
                            valid_move_found = True
                            current_point = next_point
                            measurement_point = noisy_point

                trajectory.append(current_point)
                measurement.append(measurement_point)

            trajectories.append(np.array(trajectory))
            measurements.append(np.array(measurement))

        return np.array(trajectories), np.array(measurements)

    def generate_fixed_length_trajectories(self):
        x_values = np.arange(self.range_x[0], self.range_x[1], self.step[0])
        y_values = np.arange(self.range_y[0], self.range_y[1], self.step[1])

        grid_points = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

        trajectories = []
        measurements = []

        for _ in range(self.num_trajectories):
            current_point = grid_points[np.random.choice(len(grid_points))]
            trajectory = []
            measurement = []

            for _ in range(self.trajectory_length):
                moves = [
                    [self.step[0], 0],  # Right
                    [-self.step[0], 0], # Left
                    [0, self.step[1]],  # Up
                    [0, -self.step[1]],  # Down
                    [self.step[0], self.step[1]],  # Diagonal Right-Up
                    [-self.step[0], self.step[1]], # Diagonal Left-Up
                    [self.step[0], -self.step[1]], # Diagonal Right-Down
                    [-self.step[0], -self.step[1]] # Diagonal Left-Down
                ]

                valid_move_found = False
                while not valid_move_found:
                    move = moves[np.random.choice(len(moves))]
                    next_point = current_point + move
                    noisy_point = ((next_point + np.random.normal(loc=0, scale=self.measurement_noise, size=(2))) // self.step) * self.step

                    if self.range_x[0] <= next_point[0] < self.range_x[1] and self.range_y[0] <= next_point[1] < self.range_y[1]:
                        if self.range_x[0] <= noisy_point[0] < self.range_x[1] and self.range_y[0] <= noisy_point[1] < self.range_y[1]:
                            valid_move_found = True
                            current_point = next_point
                            measurement_point = noisy_point

                trajectory.append(current_point)
                measurement.append(measurement_point)

            trajectories.append(np.array(trajectory))
            measurements.append(np.array(measurement))

        return np.array(trajectories), np.array(measurements)

    def create_data_loaders(self, base_path, batch_size, flag_flattend, validation_split=0.1):
        if self.n_modes is None:
            data_path = os.path.join(base_path, f'r2_toy_dataset_traj{self.num_trajectories}_length{self.trajectory_length}_noise{self.measurement_noise}_step{self.step}_flattend{flag_flattend}.pt')
        else:
            print('Multimodal')
            data_path = os.path.join(base_path, f'r2_toy_dataset_traj{self.num_trajectories}_length{self.trajectory_length}_noise{self.measurement_noise}_step{self.step}_modes{self.n_modes}_flattend{flag_flattend}.pt')
        if not os.path.exists(data_path):
            if self.n_modes is None:
                true_trajectories, true_measurements = self.generate_fixed_length_trajectories()
            else:
                print('Multimodal')
                true_trajectories, true_measurements = self.generate_multimodal_trajectories()
            measurements_torch = torch.from_numpy(true_measurements).type(torch.FloatTensor)
            ground_truth_torch = torch.from_numpy(true_trajectories).type(torch.FloatTensor)
            print(measurements_torch.shape, ground_truth_torch.shape)
            if flag_flattend:
                ground_truth_torch = torch.flatten(ground_truth_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
                measurements_torch = torch.flatten(measurements_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
                
            dataset = torch.utils.data.TensorDataset(ground_truth_torch, measurements_torch)
            torch.save(dataset, data_path)
        else:
            print('Loading Data')
            dataset = torch.load(data_path)
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, shuffle=False)
        
        return train_loader, val_loader
