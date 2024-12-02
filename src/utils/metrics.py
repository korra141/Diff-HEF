import torch

def kl_divergence( p, q):

    """
    Calculate the KL divergence between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return abs(torch.mean(torch.sum(p * torch.log(p / q), dim=(1, 2))))

def mean_absolute_error(p, q):
    """
    Calculate the Mean Absolute Error (MAE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    return torch.mean(torch.abs(p - q))

def root_mean_square_error(p, q):
    """
    Calculate the Root Mean Square Error (RMSE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    return torch.sqrt(torch.mean((p - q) ** 2))