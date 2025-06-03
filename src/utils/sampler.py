import torch
from typing import Tuple

def se2_grid_samples_torch(
    batch_size,
    size: Tuple[int, int, int] = (5, 5, 5),
    lower_bound: float = -0.5,
    upper_bound: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate grid samples in SE(2) space.

    Args:
        size (Tuple[int, int, int]): Grid size for (x, y, theta) dimensions.
        lower_bound (float): Lower bound for x and y dimensions.
        upper_bound (float): Upper bound for x and y dimensions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - poses: Flattened grid of SE(2) poses (N, 3), where N is the total number of grid points.
            - X: Meshgrid tensor for the x dimension.
            - Y: Meshgrid tensor for the y dimension.
            - T: Meshgrid tensor for the theta dimension.
    """
    xs = torch.linspace(lower_bound, upper_bound, size[0]+1, dtype=torch.float64)[:-1]
    ys = torch.linspace(lower_bound, upper_bound, size[1]+1, dtype=torch.float64)[:-1]
    ts = torch.linspace(0, 2*torch.pi, size[2]+1, dtype=torch.float64)[:-1]
    

    X, Y, T = torch.meshgrid(xs, ys, ts, indexing='ij')

    # Flatten the grid and stack into poses
    poses = torch.stack((X.flatten(), Y.flatten(), T.flatten()), dim=-1)
    
    poses_batch = poses.unsqueeze(0).repeat((batch_size,1, 1))

    return poses_batch, X, Y, T
