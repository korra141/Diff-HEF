import torch 
import numpy as np 
import pdb
from scipy.stats import norm

class SE2Group:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta
  def __add__(self, other):
    x = self.x + other.x * np.cos(self.theta) - other.y * np.sin(self.theta)
    y = self.y + other.y * np.cos(self.theta) + other.x * np.sin(self.theta)
    theta = self.theta + other.theta
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return SE2Group(x, y, theta)
  def parameters(self):
    self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([self.x, self.y, self.theta])
  @classmethod
  def from_parameters(cls, x, y, theta):
    return cls(x, y, theta)

class SE2SimpleSimulator:

  def __init__(self, start, step, measurement_noise, motion_noise, samples):
    self.position = start
    self.step = step
    self.motion_noise = motion_noise
    self.measurement_noise = measurement_noise
    self.beacons = np.array(
            [[0, 0.1],
             [0, 0.05],
             [0, 0.0],
             [0, -0.05],
             [0, -0.1]])
    self.beacon_idx = 0
    self.samples = samples

  def motion(self):
    # pdb.set_trace()
    self.position = self.position + self.step
    noisy_prediction = self.step.parameters() + np.random.randn(3) * self.motion_noise
    # noisy_prediction[2] = (noisy_prediction[2] + np.pi) % (2*np.pi) - np.pi
    # noisy_prediction[2] = (noisy_prediction[2]) % (2*np.pi)
    # self.position.theta = (self.position.theta + np.pi) % (2*np.pi) - np.pi
    # self.position.theta = self.position.theta  % (2*np.pi)
    return self.position.parameters() , noisy_prediction

  def measurement(self):

    self._update_beacon_idx()
    range_beacon = self.beacons[self.beacon_idx, :]
    self.range_measurement = np.linalg.norm(self.position.parameters()[0:2] - range_beacon)
    self.range_measurement += np.random.normal(0.0, self.measurement_noise, 1).item()
    dist = np.linalg.norm(range_beacon - self.samples[0,:, 0:2], axis=-1)
    range_prob = norm(dist, self.measurement_noise).pdf(self.range_measurement)
    range_ll = np.log(range_prob + 1e-8)
    return range_ll, self.range_measurement

#   def measurement(self):
#     measurement_ = self.position.parameters() + np.random.randn(3) * self.measurement_noise
#     measurement_[0:2] = (measurement_[0:2] + 0.5) % 1.0 - 0.5
#     measurement_[2] = (measurement_[2]  + np.pi ) % (2*np.pi) - np.pi
#     return measurement_

  def _update_beacon_idx(self) -> None:
    """
    Update beacon index, and cycle back to 0 if need be.
    """
    self.beacon_idx += 1
    if self.beacon_idx >= self.beacons.shape[0]:
        self.beacon_idx = 0

def random_start_pose():
    """
    Generate a random start pose within the specified bounds.
    :return: A random pose [x, y, theta] in [-0.5, 0.5] for x, y and [0, 2pi] for theta.
    """
    x = np.random.uniform(-0.1, 0.1)
    y = np.random.uniform(-0.2, 0)
    theta = np.random.uniform(0, np.pi/2)
    return SE2Group(x, y, theta)

def generate_bounded_se2_dataset(
    num_trajectories,
    trajectory_length,
    step_motion,
    motion_noise,
    measurement_noise,
    samples,
    batch_size,
    validation_split,
    test_split, 
    start_pose = None
):
    """
    Generates a dataset of SE2 trajectories and corresponding measurements within bounded space.

    :param num_trajectories: Number of trajectories to generate.
    :param trajectory_length: Number of steps per trajectory.
    :param step_motion: Step motion [dx, dy, dtheta].
    :param motion_noise: Noise for motion [x, y, theta].
    :param measurement_noise: Noise for measurements [x, y, theta].
    :param output_file: File to save the generated dataset.
    """
    true_trajectories = np.ndarray((num_trajectories, trajectory_length, 3))
    # measurements_density = np.ndarray((num_trajectories, trajectory_length - 1,samples.shape[1]))
    measurements = np.ndarray((num_trajectories, trajectory_length - 1, 1))
    noisy_control = np.ndarray((num_trajectories, trajectory_length - 1, 3))
    beacon_idx = np.ndarray((num_trajectories, trajectory_length - 1, 1))


    for traj_id in range(num_trajectories):
        # Initialize simulator with a random start pose
        if start_pose is None:
          # print("Generating random start pose")
          start_pose = random_start_pose()
        
        simulator = SE2SimpleSimulator(
            start=start_pose,
            step=step_motion,
            measurement_noise=measurement_noise,
            motion_noise=motion_noise,
            samples = samples
        )

        true_trajectories[traj_id, 0, :] = start_pose.parameters()

        for step in range(trajectory_length-1):
            # Simulate motion
            motion, noisy_step = simulator.motion()
            true_trajectories[traj_id, step + 1, :] = motion
            noisy_control[traj_id, step, :] = noisy_step

            # Simulate measurement
            measurements_density_, measurements_ = simulator.measurement()
            measurements[traj_id, step] = measurements_
            # measurements_density[traj_id, step] = measurements_density_
            beacon_idx[traj_id, step] = simulator.beacon_idx

            # # Check bounds and reset position if out of bounds
            # current_pose = simulator.position.parameters()
            # if not (-0.5 <= current_pose[0] <= 0.5 and -0.5 <= current_pose[1] <= 0.5):
            #     simulator.position = random_start_pose()

    measurements_torch = torch.from_numpy(measurements).type(torch.DoubleTensor)
    # measurements_density_torch = torch.from_numpy(measurements_density).type(torch.DoubleTensor)
    ground_truth_torch = torch.from_numpy(true_trajectories).type(torch.DoubleTensor)
    control_torch = torch.from_numpy(noisy_control).type(torch.DoubleTensor)
    beacon_idx_torch = torch.from_numpy(beacon_idx).type(torch.long)
    # ground_truth_torch = torch.flatten(ground_truth_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
    # measurements_torch = torch.flatten(measurements_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
    dataset = torch.utils.data.TensorDataset(ground_truth_torch, measurements_torch, control_torch, beacon_idx_torch)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    val_split = int(np.floor(validation_split * (dataset_size - test_split)))

    np.random.shuffle(indices)
    test_indices = indices[:test_split]
    remaining_indices = indices[test_split:]
    val_indices = remaining_indices[:val_split]
    train_indices = remaining_indices[val_split:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False,  num_workers=4, pin_memory=True)
  
    return train_loader, val_loader, test_loader
