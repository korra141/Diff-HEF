import torch
import pdb

def kl_divergence( p, q):

    """
    Calculate the KL divergence between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return abs(torch.mean(torch.sum(p * torch.log(p / q), dim=(1, 2))))

def kl_divergence_s1( p, q):

    """
    Calculate the KL divergence between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return abs(torch.mean(torch.sum(p * torch.log(p / q),dim=-1)))

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
    return torch.sqrt(torch.mean((p - q) ** 2,))


def root_mean_square_error_s1(p, q):
    """
    Calculate the Root Mean Square Error (RMSE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    return torch.mean(torch.sqrt(torch.mean((p - q) ** 2,dim=(-1))))

def wasserstein_distance(p, q):
    """
    Calculate the Wasserstein distance (Earth Mover's Distance) between two 1D distributions p and q in a batch.
    Both p and q should be 2D torch tensors of the same shape, where the first dimension is the batch size.
    """
    p_sorted, _ = torch.sort(p, dim=-1)
    q_sorted, _ = torch.sort(q, dim=-1)
    return torch.sum(torch.abs(p_sorted - q_sorted), dim=-1).mean()

def predicted_residual_error(p, q, bins=10):
    """
    Calculate the predicted residual error between two tensors p and q and store the frequency of the values in bins.
    Both p and q should be torch tensors of the same shape.
    """
    residual_error = p - q
    hist = torch.histc(residual_error, bins=bins)
    min_val, max_val = torch.min(residual_error), torch.max(residual_error)
    bin_edges = torch.linspace(min_val, max_val, 20 + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return residual_error, hist, bin_centers, bin_edges


def expected_calibration_error_continuous(y_true, y_pred_cdf, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for continuous distributions.
    
    Args:
        y_true (torch.Tensor): Observed data points of shape (n_samples,).
        y_pred_cdf (torch.Tensor): Predicted CDF values corresponding to y_true, shape (n_samples,).
        n_bins (int): Number of bins for the calibration evaluation.
    
    Returns:
        float: The ECE score.
    """
    # Define bin edges based on observed data
    min_val, max_val = torch.min(y_true), torch.max(y_true)
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1)

    # Initialize ECE and bin population
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        # Get the bin range
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        
        # Find samples in the current bin
        in_bin = (y_true >= lower_edge) & (y_true < upper_edge)
        n_in_bin = torch.sum(in_bin).item()
        
        if n_in_bin == 0:
            continue  # Skip empty bins
        
        # Compute empirical CDF for the bin
        empirical_cdf = torch.mean((y_true <= upper_edge).float())

        # Compute predicted CDF for the bin
        predicted_cdf = torch.mean(y_pred_cdf[in_bin])
        
        # Compute the bin's contribution to ECE
        ece += n_in_bin / total_samples * torch.abs(predicted_cdf - empirical_cdf)

    return torch.mean(ece)

# # Example usage
# # Observed data (true values)
# y_true = torch.tensor([0.1, 0.4, 0.5, 0.8, 0.9])

# # Predicted CDF values corresponding to y_true
# # These could come from a probabilistic model, e.g., a normal distribution
# y_pred_cdf = torch.tensor([0.2, 0.5, 0.6, 0.7, 0.95])

# # Compute ECE
# ece_score = expected_calibration_error_continuous(y_true, y_pred_cdf, n_bins=5)
# print(f"Expected Calibration Error (ECE): {ece_score}")



def compute_cdf_from_pdf(pdf):
    """
    Compute the CDF from the given PDF values.

    Args:
        pdf (torch.Tensor): PDF values over the support, shape (n_samples,).

    Returns:
        torch.Tensor: CDF values over the same support, shape (n_samples,).
    """
    # Ensure the PDF is normalized
    pdf_sum = torch.sum(pdf)
    if not torch.isclose(pdf_sum, torch.tensor(1.0), atol=1e-6):
        pdf = pdf / pdf_sum

    # Compute the cumulative sum to get the CDF
    cdf = torch.cumsum(pdf, dim=0)
    return cdf

# # Example PDF values
# pdf = torch.tensor([0.1, 0.2, 0.3, 0.4])

# # Compute the CDF
# cdf = compute_cdf_from_pdf(pdf)
# print("PDF:", pdf)
# print("CDF:", cdf)



