import torch
import numpy as np

class SE2Group:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta
  def __add__(self, other):
    x = self.x + other.x * np.cos(self.theta) - other.y * np.sin(self.theta)
    y = self.y + other.y * np.cos(self.theta) + other.x * np.sin(self.theta)
    theta = self.theta + other.theta
    return SE2Group(x, y, theta)
  def parameters(self):
    return np.array([self.x, self.y, self.theta])
  @classmethod
  def from_parameters(cls, x, y, theta):
    return cls(x, y, theta)

class SE2SimpleSimulator:

  def __init__(self, start, step, measurement_noise, motion_noise):
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

  def motion(self):

    self.position = self.position + self.step
    noise = np.random.normal(loc=0.0, scale=self.motion_noise, size=3)
    noisy_prediction = self.step.parameters() + noise
    noisy_prediction[2] = (noisy_prediction[2] + np.pi) % (2*np.pi) - np.pi
    self.position.theta = (self.position.theta + np.pi) % (2*np.pi) - np.pi
    positions = self.position.parameters()
    positions[0:2] = np.sign(positions [0:2]) * np.minimum(abs(positions [0:2]), (1 - abs(positions [0:2])))
    self.position = SE2Group(*positions)
    if np.any(positions[0:2] < -0.5) or np.any(positions[0:2] > 0.5):
      import pdb; pdb.set_trace()
    return positions  , noisy_prediction
#   def measurement(self):

#     self._update_beacon_idx()
#     range_beacon = self.beacons[self.beacon_idx, :]
#     # Observation z_t
#     self.range_measurement = np.linalg.norm(self.position.parameters()[0:2] - range_beacon)
#     # Jitter range measurement with noise
#     self.range_measurement += np.random.normal(0.0, self.measurement_noise, 1).item()

#     return self.range_measurement

  def measurement(self):
    # Generate a noise vector with mean 0 and different std deviations
    noise = np.random.normal(loc=0.0, scale=self.measurement_noise)
    measurement_ = self.position.parameters() + noise
    measurement_temp = np.ones(3)
    measurement_temp[0:2] = np.sign(measurement_[0:2]) * np.minimum(abs(measurement_[0:2]), (1 - abs(measurement_[0:2])))
    if np.any(measurement_temp[0:2] < -0.5) or np.any(measurement_temp[0:2] > 0.5):
      import pdb; pdb.set_trace()
    measurement_temp[2] = (measurement_[2]  + np.pi ) % (2*np.pi) - np.pi
    return measurement_temp

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
    x = np.random.uniform(-0.2, 0.2)
    y = np.random.uniform(-0.2, 0.2)
    theta = np.random.uniform(-np.pi, np.pi)
    return SE2Group(x, y, theta)

def generate_bounded_se2_dataset(
    num_trajectories,
    trajectory_length,
    step_motion,
    motion_noise,
    measurement_noise,
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
    measurements = np.ndarray((num_trajectories, trajectory_length,3))
    noisy_control = np.ndarray((num_trajectories, trajectory_length, 3))


    for traj_id in range(num_trajectories):
        # Initialize simulator with a random start pose
        start_pose = random_start_pose()
        simulator = SE2SimpleSimulator(
            start=start_pose,
            step=step_motion,
            measurement_noise=measurement_noise,
            motion_noise=motion_noise,
        )

        for step in range(trajectory_length):
            # Simulate motion
            motion, noisy_step = simulator.motion()
            true_trajectories[traj_id, step, :] = motion
            noisy_control[traj_id, step, :] = noisy_step

            # Simulate measurement
            measurements_ = simulator.measurement()
            measurements[traj_id, step] = measurements_


    return true_trajectories, noisy_control, measurements

def create_dataloaders(
    num_trajectories,
    trajectory_length,
    step_motion,
    motion_noise,
    measurement_noise,
    batch_size,
    validation_split
):

    step_motion = SE2Group(*step_motion)
    # Generate dataset
    true_trajectories, noisy_control, measurements = generate_bounded_se2_dataset(
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        step_motion=step_motion,
        motion_noise=motion_noise,
        measurement_noise=measurement_noise,
    )

    measurements_torch = torch.from_numpy(measurements).type(torch.FloatTensor)
    ground_truth_torch = torch.from_numpy(true_trajectories).type(torch.FloatTensor)
    ground_truth_torch = torch.flatten(ground_truth_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
    measurements_torch = torch.flatten(measurements_torch, start_dim=0, end_dim=1).type(torch.FloatTensor)
    dataset = torch.utils.data.TensorDataset(ground_truth_torch, measurements_torch)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, shuffle=False, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True, shuffle=False, pin_memory=False)

    return train_loader, val_loader
