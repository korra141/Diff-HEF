import torch 
import numpy as np
import math
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
from src.distributions.S1.BetaDistribution import BetaDistribution

def generating_data_S1_unimodal(base_path,batch_size, n_samples, trajectory_length, measurement_noise, step=0.1, shuffle_flag=True):
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
    data_path = os.path.join(base_path, f's1_toy_dataset_flattened_{n_samples * trajectory_length}_noise{measurement_noise}_step{step}.pt')
    if not os.path.exists(data_path):
        print('Generating Data at:', data_path) 
        starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        true_trajectories = np.ndarray((n_samples, trajectory_length))
        measurements = np.ndarray((n_samples, trajectory_length))

        for i in range(n_samples):
            # Generate a circular trajectory with a random starting position.
            initial_angle = starting_positions[i]
            trajectory = initial_angle + np.arange(trajectory_length) * step
            true_trajectories[i] = trajectory

            # Add Gaussian noise to the measurements.
            measurements[i] = (trajectory + np.random.normal(0, measurement_noise, trajectory_length))% (2 * np.pi)

        measurements_ = torch.from_numpy(measurements % (2 * np.pi))
        ground_truth_ = torch.from_numpy(true_trajectories % (2 * np.pi))
        ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
        torch.save(train_dataset, data_path)
    else:
        print('Loading Data')
        train_dataset = torch.load(data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)
    return train_loader

def generating_data_S1_unimodal_beta(base_path,batch_size, n_samples, trajectory_length, alpha=2, beta=5, step=0.1, shuffle_flag=True):
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
    data_path = os.path.join(base_path, f's1_toy_dataset_flattened_{n_samples * trajectory_length}_noisebeta{alpha}_{beta}_step{step}.pt')
    if not os.path.exists(data_path):
        print('Generating Data at:', data_path) 
        starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        true_trajectories = np.ndarray((n_samples, trajectory_length))
        measurements = np.ndarray((n_samples, trajectory_length))
        beta_distribution = BetaDistribution(alpha, beta)
        for i in range(n_samples):
            # Generate a circular trajectory with a random starting position.
            initial_angle = starting_positions[i]
            trajectory = initial_angle + np.arange(trajectory_length) * step
            true_trajectories[i] = trajectory
            # Add Gaussian noise to the measurements.
            measurements[i] = (trajectory + beta_distribution.sample(size=trajectory_length))% (2 * np.pi)
        measurements_ = torch.from_numpy(measurements % (2 * np.pi))
        ground_truth_ = torch.from_numpy(true_trajectories % (2 * np.pi))
        ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
        torch.save(train_dataset, data_path)
    else:
        print('Loading Data')
        train_dataset = torch.load(data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)
    return train_loader
def heteroscedastic_noise(x, min_noise, max_noise):
   # Normalize x 
   x = x % (2 * np.pi)
   x = x / (2 * np.pi)
   scale = x**2 / (x**2 + (1-x)**2)
   return min_noise + scale * (max_noise - min_noise)

def mean_var(mu, sigma):
    epsilon = 1e-3  
    kappa = mu * (1 - mu) / (sigma ** 2) - 1
    # alpha = mu * kappa
    # beta = (1 - mu) * kappa
    alpha = np.clip(mu * kappa, a_min=1 + epsilon, a_max=None)
    beta = np.clip((1 - mu) * kappa, a_min=1 + epsilon, a_max=None)
    return alpha, beta

def generating_data_S1_heteroscedastic_beta(base_path, args, batch_size, n_samples, trajectory_length, step=0.1, shuffle_flag=True):
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
    data_path = os.path.join(base_path, f's1_toy_dataset_hetero_flattened_{n_samples * trajectory_length}_noisebeta_{args.alpha}{args.beta}_{args.measurement_noise_min}_{args.measurement_noise_max}_step{step}.pt')
    if not os.path.exists(data_path):
        print('Generating Data at:', data_path) 
        starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        true_trajectories = np.ndarray((n_samples, trajectory_length))
        measurements = np.ndarray((n_samples, trajectory_length))

        for i in range(n_samples):
            # Generate a circular trajectory with a random starting position.
            initial_angle = starting_positions[i]
            trajectory = initial_angle + np.arange(trajectory_length) * step
            true_trajectories[i] = trajectory

            # Add Gaussian noise to the measurements.
            # noise = [np.random.normal(0, heteroscedastic_noise(x, args.measurement_noise_min, args.measurement_noise_max)) for x in trajectory]
            mu = args.alpha/(args.alpha + args.beta)
            noise = [BetaDistribution(*mean_var(mu, heteroscedastic_noise(x, args.measurement_noise_min, args.measurement_noise_max))).sample(size=1).item() for x in trajectory]  
            # beta_distribution = BetaDistribution(parameters[:,0], parameters[:,1])
            # noise = beta_distribution.sample(size=trajectory_length)
            # print(noise[0].shape)
            measurements[i] = (trajectory + noise) % (2 * np.pi)

        measurements_ = torch.from_numpy(measurements % (2 * np.pi))
        ground_truth_ = torch.from_numpy(true_trajectories % (2 * np.pi))
        ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
        torch.save(train_dataset, data_path)
    else:
        print('Loading Data')
        train_dataset = torch.load(data_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)
    return train_loader

def generating_data_S1_heteroscedastic(base_path, args, batch_size, n_samples, trajectory_length, measurement_noise, step=0.1, shuffle_flag=True):
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
    data_path = os.path.join(base_path, f's1_toy_dataset_hetero_flattened_{n_samples * trajectory_length}_noise_{args.measurement_noise_min}_{args.measurement_noise_max}_step{step}.pt')
    if not os.path.exists(data_path):
        print('Generating Data at:', data_path) 
        starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        true_trajectories = np.ndarray((n_samples, trajectory_length))
        measurements = np.ndarray((n_samples, trajectory_length))

        for i in range(n_samples):
            # Generate a circular trajectory with a random starting position.
            initial_angle = starting_positions[i]
            trajectory = initial_angle + np.arange(trajectory_length) * step
            true_trajectories[i] = trajectory

            # Add Gaussian noise to the measurements.
            noise = [np.random.normal(0, heteroscedastic_noise(x, args.measurement_noise_min, args.measurement_noise_max)) for x in trajectory]
            measurements[i] = (trajectory + noise) % (2 * np.pi)

        measurements_ = torch.from_numpy(measurements % (2 * np.pi))
        ground_truth_ = torch.from_numpy(true_trajectories % (2 * np.pi))
        ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
        torch.save(train_dataset, data_path)
    else:
        print('Loading Data')
        train_dataset = torch.load(data_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)
    return train_loader



def sample_mm(n_modes, mean, std, n_samples):
    weights = np.ones(n_modes) / n_modes
    selected_mode = np.random.choice(range(n_modes), size=n_samples, p=weights) # n_samples
    noise_point = np.random.normal(loc=mean[selected_mode], scale=std[selected_mode])
    return noise_point

def generating_data_S1_multimodal(measurement_noise, mean_offset,n_modes,base_path, batch_size, n_samples, trajectory_length, step=0.1, shuffle_flag=True, flattend=True):
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
  data_path = os.path.join(base_path, f's1_toy_dataset_flattened_{flattend}_{n_samples * trajectory_length}_mean{mean_offset:.4f}_mm_noise{measurement_noise}_step{step}.pt')
  if not os.path.exists(data_path):
    print('Generating Data at:', data_path) 
    starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    true_trajectories = np.ndarray((n_samples, trajectory_length))
    measurements = np.ndarray((n_samples, trajectory_length))
    for i in range(n_samples):
      # Generate a circular trajectory with a random starting position.
      initial_angle = starting_positions[i]
      trajectory = initial_angle + np.arange(trajectory_length) * step
      true_trajectories[i] = trajectory % (2 * np.pi)

      mean = np.linspace(-mean_offset / 2, mean_offset / 2, n_modes)
          
      # Add Gaussian noise to the measurements.
      noise = sample_mm(n_modes, mean, measurement_noise, trajectory.shape[0])
      # noise = multimodal_gaussian_noise(measurement_noise,mean_offset,bin_prob,trajectory_length)
      
      measurements[i] = (trajectory + noise) % (2 * np.pi)

    measurements_ = torch.from_numpy(measurements)
    ground_truth_ = torch.from_numpy(true_trajectories)
    if flattend:
        ground_truth_flatten = torch.flatten(ground_truth_)[:, None].type(torch.FloatTensor)
        measurements_flatten = torch.flatten(measurements_)[:, None].type(torch.FloatTensor)
        train_dataset = torch.utils.data.TensorDataset(ground_truth_flatten, measurements_flatten)
    else:
        train_dataset = torch.utils.data.TensorDataset(ground_truth_.unsqueeze(-1), measurements_.unsqueeze(-1))
    torch.save(train_dataset, data_path)
  else:
    print('Loading Data')
    train_dataset = torch.load(data_path)
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle_flag)

  return train_loader