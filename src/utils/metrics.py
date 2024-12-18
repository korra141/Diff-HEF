import torch
import pdb
import math

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

def absolute_error_s1(p,q):
    """
    Calculate the Absolute Error between two distributions p and q.
    Both p and q should be torch tensors of the same shape.
    """
    error = abs(p - q)
    stacked_tensors = torch.stack((error, torch.ones_like(error)*2*math.pi - error), dim=-1)
    # Find the minimum values across the last dimension
    min_values, min_indices = torch.min(stacked_tensors, dim=-1)
    return min_values
def mean_absolute_error(p, q):
    """
    Calculate the Mean Absolute Error (MAE) between two tensors p and q.
    Both p and q should be torch tensors of the same shape.
    """
    min_values = absolute_error_s1(p, q)
    return torch.mean(min_values)

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
    min_values = absolute_error_s1(p, q)
    return torch.mean(torch.sqrt(torch.mean(min_values ** 2,dim=(-1))))

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
    residual_error = absolute_error_s1(p, q)
    if torch.isnan(residual_error).any():
        print("residual error is nan")
    hist = torch.histc(residual_error, bins=bins)
    min_val, max_val = torch.min(residual_error), torch.max(residual_error)
    bin_edges = torch.linspace(min_val, max_val, 20 + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return residual_error, hist, bin_centers, bin_edges


def expected_calibration_error_continuous(y_true, y_pred_cdf, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for continuous distributions.
    
    Args:
        y_true (torch.Tensor): Values of the true distribution pdf
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
    total_samples = y_true.size(1)
    
    for i in range(n_bins):
        # Get the bin range
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        # pdb.set_trace()
        # Find samples in the current bin
        in_bin = (y_true >= lower_edge) & (y_true < upper_edge)
        n_in_bin = torch.sum(in_bin,dim=1)
        
        # if n_in_bin == 0:
        #     continue  # Skip empty bins
        
        # Compute empirical CDF for the bin
        empirical_cdf = torch.mean((y_true <= upper_edge).float(), dim=1)

        # Compute predicted CDF for the bin
        predicted_cdf = torch.mean(torch.where(in_bin, y_pred_cdf, torch.tensor(0)), dim=1)
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

def expected_calibration_error(predicted_distribution, true_distribution, M=5):
    """
    Computes the Expected Calibration Error (ECE) between the predicted and true distributions.
    ECE is a scalar measure of how well the predicted probabilities are calibrated with respect to the true outcomes.
    It is calculated by partitioning the predictions into M bins and computing the weighted average of the absolute 
    difference between the accuracy and confidence of each bin.
    Args:
        predicted_distribution (torch.Tensor): A tensor of predicted probabilities with shape (batch_size, num_samples).
        true_distribution (torch.Tensor): A tensor of true probabilities with shape (batch_size, num_samples).
        M (int, optional): The number of bins to partition the predicted probabilities into. Default is 5.
    Returns:
        torch.Tensor: A tensor containing the ECE for each batch element.
    """
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    total_samples = predicted_distribution.shape[-1]
    batch_size = predicted_distribution.shape[0]
    ece = torch.zeros(batch_size)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        in_bin_pred = torch.logical_and(predicted_distribution > bin_lower, predicted_distribution <= bin_upper)
        # Identify samples where the true PDF is also within the bin
        in_bin_true = torch.logical_and(true_distribution > bin_lower, true_distribution <= bin_upper)
        
        # Determine the samples that satisfy both conditions
        matched_samples = torch.logical_and(in_bin_pred, in_bin_true)
        n_in_bin = torch.sum(in_bin_pred,dim=1)
        if torch.any(n_in_bin > 0):
            non_zero_bins = n_in_bin > 0
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = torch.where(non_zero_bins, matched_samples.sum(dim=-1) / n_in_bin, torch.tensor(0.0))
            # get the average confidence of bin m: conf(Bm)
            masked_tensor = torch.where(in_bin_pred, predicted_distribution, torch.tensor(0.0))
            avg_confidence_in_bin = torch.where(non_zero_bins, masked_tensor.sum(dim=-1) / n_in_bin, torch.tensor(0.0))
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.where(non_zero_bins, abs(avg_confidence_in_bin - accuracy_in_bin) * (n_in_bin / total_samples), torch.tensor(0.0))

    return torch.mean(ece)

def expected_calibration_error_check(predicted_distribution, true_distribution, M=5):
    """
    Computes the Expected Calibration Error (ECE) between the predicted and true distributions.
    """
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    batch_size, num_samples = predicted_distribution.shape
    ece = torch.zeros(batch_size, device=predicted_distribution.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Mask for samples in the current bin
        in_bin_pred = (predicted_distribution > bin_lower) & (predicted_distribution <= bin_upper)
        in_bin_true = (true_distribution > bin_lower) & (true_distribution <= bin_upper)
        matched_samples = in_bin_pred & in_bin_true
        
        # Count samples in the bin
        n_in_bin = in_bin_pred.sum(dim=-1)
        
        # Avoid division by zero
        non_zero_bins = n_in_bin > 0
        
        # Calculate accuracy and confidence
        accuracy_in_bin = torch.zeros_like(ece)
        avg_confidence_in_bin = torch.zeros_like(ece)
        
        if non_zero_bins.any():
            accuracy_in_bin[non_zero_bins] = matched_samples.sum(dim=-1)[non_zero_bins] / n_in_bin[non_zero_bins]
            masked_tensor = torch.where(in_bin_pred, predicted_distribution, torch.tensor(0.0))
            avg_confidence_in_bin[non_zero_bins] = masked_tensor.view(batch_size, -1).mean(dim=-1)[non_zero_bins]
        
        # Weighted absolute difference
        ece += torch.where(
            non_zero_bins,
            torch.abs(avg_confidence_in_bin - accuracy_in_bin) * (n_in_bin / num_samples),
            torch.tensor(0.0, device=ece.device)
        )
    
    return ece.mean()
def sharpness_discrete(pred_probs):
    """
    Compute sharpness for discrete distributions based on entropy.
    
    Parameters:
    - pred_probs: torch.Tensor, predicted probabilities (N, K), where K is the number of classes.
    
    Returns:
    - sharpness: float, the average sharpness metric.
    """
    entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-9), dim=1)  # Add epsilon to avoid log(0)
    return torch.mean(entropy)

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



