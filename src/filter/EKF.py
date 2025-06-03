import torch
import pdb
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.metrics import absolute_error_s1, error_s1
from src.utils.group import SE2Group

class ExtendedKalmanFilter:
    def __init__(self,initial_covariance, process_cov):
        self.P = torch.tensor(initial_covariance) # Covariance matrix
        self.Q = torch.tensor(process_cov)  # Process noise covariance
    
    def set_initial_state(self, x):
        """
        Set the initial state of the filter.
        x: Initial state
        """
        self.x = x
    
    def set_step(self, step):
        """
        Set the step of the filter.
        step: Step of the filter
        """
        self.step = step

    def predict(self):
        """
        Prediction step.
        f: Non-linear state transition function
        F: Jacobian of the state transition function
        u: Control input (optional)
        """
        positive_angle = torch.ones_like(self.x) * (2*math.pi)
        self.x = (self.x + self.step) % (2*math.pi)
        if torch.any(self.x < 0):
            pdb.set_trace()
        self.P =  self.P + self.Q
    
    # def check_left_or_right(self,z,z_pred):
    #     check = z - z_pred
    #     left = torch.where(check < 0 and check.abs() < math.pi , True, False)
    #     return left

    def update(self, z, R):
        """
        Update step of EKF.
        :param z: Measurement (numpy array)
        :param H: Jacobian of measurement model (numpy array)
        :param h: Nonlinear measurement function (callable, h(x))
        """
        z_pred = self.x
        # jacobian_h = torch.ones_like(z_pred)
        positive_angle = torch.ones_like(z_pred) * (2*math.pi)
        # Innovation (residual)s
        y = error_s1(z,z_pred)
        # Innovation covariancet
        S = self.P + R

        # Kalman gain
        K = self.P  / S
        
        # Update state estimate
        self.x = (self.x + K * y) % (2*math.pi) # Update state estimate
        if torch.any(self.x < 0):
            pdb.set_trace()
        # Update covariance estimate
        identity = torch.ones(K.shape)
        self.P = (identity - K) * self.P



import torch
from typing import Tuple


class RangeEKF:
    def __init__(self,
                 prior: torch.Tensor,
                 prior_cov: torch.Tensor):
        """
        Extended Kalman Filter for range-based localization.
        
        Args:
            prior: Prior pose as a tensor of dimension (3,)
            prior_cov: Covariance noise for the prior distribution of dimension (3, 3)
        """
        self.pose: SE2Group = SE2Group.from_batched_parameters(prior)
        self.state_cov: torch.Tensor = prior_cov.clone()

    def prediction(self,
                   step: torch.Tensor,
                   step_cov: torch.Tensor) -> None:
        """
        Prediction step of EKF.
        
        Args:
            step: Motion step (relative displacement) of dimension (3,)
            step_cov: Covariance matrix of prediction step of dimension (3, 3)
        """
        # Get group representation of current pose
        pose = self.pose.parameters()
        batch_size = step.shape[0]
        
        g = torch.zeros((batch_size, 3, 3), dtype=torch.float64, device=pose.device)
    
        # Fill in the fixed values
        g[:, 0, 0] = 1.0
        g[:, 1, 1] = 1.0
        g[:, 2, 2] = 1.0
        
        # Fill in the variable elements
        g[:, 0, 2] = -step[:, 0] * torch.sin(pose[:, 2]) - step[:, 1] * torch.cos(pose[:, 2])
        g[:, 1, 2] = step[:, 0] * torch.cos(pose[:, 2]) - step[:, 1] * torch.sin(pose[:, 2])
    

        # Matrix representation of current pose
        step_group = SE2Group.from_batched_parameters(step)
        
        # Propagate step (mean)
        self.pose = self.pose + step_group
        temp = torch.bmm(g, self.state_cov) # [batch, 3, 3]
        self.state_cov = torch.bmm(temp, g.transpose(1, 2)) + step_cov  # [batch, 3, 3]
        return self.pose.parameters(), self.state_cov

    def update(self,
               observations: torch.Tensor,
               z_hat: torch.Tensor,
               jacobian_h: torch.Tensor,
               observations_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update step of EKF.
        
        Args:
            landmarks: Location of each UWB landmark in the map (n, 3)
            observations: Range measurements of dimension (n,)
            observations_cov: Variance of each measurement of dimension (n,)
            
        Returns:
            tuple: Normalized belief as the mean and covariance of a Gaussian distribution
        """
        # Compute mean measurements
        batch_size = observations.shape[0]
        # Need to perform batch matrix multiplication: h @ state_cov @ h^T
        jacobian_h_transposed = jacobian_h.transpose(1, 2)  # [batch, 3, n]
        
        # First multiply state_cov by h^T: [batch, 3, 3] @ [batch, 3, n] -> [batch, 3, n]
        temp = torch.bmm(self.state_cov, jacobian_h_transposed)
        
        # Then multiply h by temp: [batch, n, 3] @ [batch, 3, n] -> [batch, n, n]
        s = torch.bmm(jacobian_h, temp)

        # observations_cov_ = torch.tile(observations_cov.unsqueeze(0), [batch_size , 1,  1])

        s = s +  observations_cov

        # Update innovation covariance
        # s = h @ self.state_cov @ h.transpose(0, 1) + 
        
        # Compute Kalman gain
        # k = self.state_cov @ h.transpose(0, 1) @ torch.inverse(s)

        # Need to solve the system s @ k^T = h @ state_cov^T for k^T
        # Using batch solve instead of inverse for stability
        temp = torch.bmm(jacobian_h, self.state_cov.transpose(1, 2))  # [batch, n, 3]
        k_transposed = torch.linalg.solve(s, temp)  # [batch, n, 3]
        k = k_transposed.transpose(1, 2)  # [batch, 3, n]

        innovation = observations - z_hat
        pose_update = self.pose.parameters() + torch.bmm(k, innovation.unsqueeze(-1)).squeeze(-1)  # [batch, 3]

        # Normalize angles to [-π, π]
        pose_update[:, 2] = (pose_update[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Update pose (assuming SE2Group can now handle batched inputs)
        self.pose = SE2Group.from_batched_parameters(pose_update)

        identity = torch.eye(3, dtype=torch.float64, device=self.pose.parameters().device).unsqueeze(0).expand(batch_size, -1, -1)
        k_h = torch.bmm(k, jacobian_h)  # [batch, 3, 3]
        self.state_cov = torch.bmm(identity - k_h, self.state_cov)  # [batch, 3, 3]

        return self.pose.parameters(), self.state_cov

    def neg_log_likelihood(self, pose: torch.Tensor, mean, cov) -> torch.Tensor:
        """
        Evaluate posterior distribution of a multivariate Gaussian.
        
        Args:
            pose: Pose at which to evaluate the likelihood 
            mean: Mean of the Gaussian 
            cov: Covariance of the Gaussian 
            
        Returns:
            torch.Tensor: Negative log likelihood
        """

        k = mean.shape[1]
        diff = pose - mean
        if diff.shape[1] == 3:
            diff[:,  2] = (diff[:,  2] + torch.pi) % (2 * torch.pi) - torch.pi
        precision_matrix = torch.inverse(cov)
        log_det = torch.logdet(cov)
        diff_expanded = diff.unsqueeze(1) # [batch, 1, D]
        mahalanobis_dist = torch.bmm(torch.bmm(diff_expanded, precision_matrix),
        diff.unsqueeze(-1)).squeeze() # [batch, 1]
        # mahalanobis_dist = torch.sum(diff @ precision_matrix @ diff, dim=1).unsqueeze(-1)

        log_constant = k * torch.log(torch.tensor(2 * torch.pi)) + log_det 
        log_likelihood = -0.5 * (log_constant + mahalanobis_dist)
        
        return torch.mean(-log_likelihood)