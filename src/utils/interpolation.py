
import torch

def circular_pad(f, pad):
    """
    Apply circular padding to a 2D tensor.
    
    Args:
        f (torch.Tensor): Input tensor of shape (H, W).
        pad (tuple): Padding in the form (top, bottom, left, right).
    
    Returns:
        torch.Tensor: Circularly padded tensor.
    """
    top, bottom, left, right = pad

    # Pad vertically (top and bottom)
    f_padded = torch.cat([f[:, -top:], f, f[:, :bottom]], dim=1) if top or bottom else f

    # Pad horizontally (left and right)
    f_padded = torch.cat([f_padded[:, :, -left:], f_padded, f_padded[:, :, :right]], dim=2) if left or right else f_padded

    return f_padded

def bilinear_interpolate_torch_circular_padded(f, x, y):
    """
    Perform bilinear interpolation on a 2D tensor with circular boundary conditions using padding.
    
    Args:
        f (torch.Tensor): Input tensor of shape (H, W).
        x (torch.Tensor): x-coordinates to interpolate (float tensor).
        y (torch.Tensor): y-coordinates to interpolate (float tensor).
    
    Returns:
        torch.Tensor: Interpolated values at the given coordinates.
    """
    B, H, W = f.shape
    pad = (1, 1, 1, 1)  # (top, bottom, left, right)

    # Circularly pad the tensor
    f_padded = torch.nn.functional.pad(f, (1, 1, 1, 1), mode='replicate')
    
    # Adjust coordinates for the padded tensor
    x = x + 1  # Shift by the padding
    y = y + 1

    # Get integer grid points
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, f_padded.shape[2] - 1)
    x1 = torch.clamp(x1, 0, f_padded.shape[2] - 1)
    y0 = torch.clamp(y0, 0, f_padded.shape[1] - 1)
    y1 = torch.clamp(y1, 0, f_padded.shape[1] - 1)


    # Gather values from the padded tensor
    Ia = f_padded[:, y0, x0]  # Bottom-left
    Ib = f_padded[:, y1, x0]  # Top-left
    Ic = f_padded[:, y0, x1]  # Bottom-right
    Id = f_padded[:, y1, x1]  # Top-right

    # Compute weights for interpolation
    wa = (x1.float() - x) * (y1.float() - y)  # Weight for Ia
    wb = (x1.float() - x) * (y - y0.float())  # Weight for Ib
    wc = (x - x0.float()) * (y1.float() - y)  # Weight for Ic
    wd = (x - x0.float()) * (y - y0.float())  # Weight for Id

    weight_sum = wa + wb + wc + wd
    
    wa = wa / (weight_sum)
    wb = wb / (weight_sum)
    wc = wc / (weight_sum)
    wd = wd / (weight_sum)
    
    # print(f"wa shape {wa.shape}")
    # print(f"Ia shape {Ia.shape}")
    
    wa = wa.unsqueeze(0)
    wb = wb.unsqueeze(0)
    wc = wc.unsqueeze(0)
    wd = wd.unsqueeze(0)


    # Compute the interpolated result
    result = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return result

 

def bilinear_interpolate_torch_with_nan(f, x, y):
    """
    Perform bilinear interpolation on a 2D tensor with `NaN` padding in PyTorch.
    
    Parameters:
        f (torch.Tensor): Input tensor of shape (H, W).
        x (torch.Tensor): x-coordinates to interpolate (float tensor).
        y (torch.Tensor): y-coordinates to interpolate (float tensor).
    
    Returns:
        torch.Tensor: Interpolated values at the given coordinates.
    """

    # Adjust coordinates for the padded tensor
    x = x + 1
    y = y + 1

    # Get integer grid points
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, f.shape[2] - 1)
    x1 = torch.clamp(x1, 0, f.shape[2] - 1)
    y0 = torch.clamp(y0, 0, f.shape[1] - 1)
    y1 = torch.clamp(y1, 0, f.shape[1] - 1)

#     pdb.set_trace()
    # Gather values from the 2D tensor
    Ia = f[:, y0, x0]  # Bottom-left
    Ib = f[:, y1, x0]  # Top-left
    Ic = f[:, y0, x1]  # Bottom-right
    Id = f[:, y1, x1]  # Top-right

#     x0 = torch.clamp(x0, 0, f.shape[1] - 1)
#     x1 = torch.clamp(x1, 0, f.shape[1] - 1)
#     y0 = torch.clamp(y0, 0, f.shape[2] - 1)
#     y1 = torch.clamp(y1, 0, f.shape[2] - 1)

# #     pdb.set_trace()
#     # Gather values from the 2D tensor
#     Ia = f[:, x0, y0]  # Bottom-left
#     Ib = f[:, x1, y0]  # Top-left
#     Ic = f[:, x0, y1]  # Bottom-right
#     Id = f[:, x1, x1]  # Top-right
    
#     print(f"Ia {Ia}, Ib {Ib}, Ic {Ic}, Id {Id}")
    # Compute the weights for interpolation
    wa = (x1 - x) * (y1 - y)  # Weight for Ia
    wb = (x1 - x) * (y - y0)  # Weight for Ib
    wc = (x - x0) * (y1 - y)  # Weight for Ic
    wd = (x - x0) * (y - y0)  # Weight for Id
    
#     print(f"wa {wa}, wb {wb}, wc {wc}, wd {wd}")

    # Mask out NaN values
    valid_a = ~torch.isnan(Ia)
    valid_b = ~torch.isnan(Ib)
    valid_c = ~torch.isnan(Ic)
    valid_d = ~torch.isnan(Id)
    
#     print(f"valid_a {valid_a}, valid_b {valid_b}, valid_c {valid_c}, valid_d {valid_d}")
    

    # Set weights to zero where values are NaN
    wa = wa * valid_a
    wb = wb * valid_b
    wc = wc * valid_c
    wd = wd * valid_d

    # Normalize weights to handle missing data
    weight_sum = wa + wb + wc + wd
#     print("weight_sum", weight_sum)
    
    Ia = torch.where(valid_a, Ia, torch.zeros_like(Ia))
    Ib = torch.where(valid_b, Ib, torch.zeros_like(Ib))
    Ic = torch.where(valid_c, Ic, torch.zeros_like(Ic))
    Id = torch.where(valid_d, Id, torch.zeros_like(Id))
    
    wa = wa / (weight_sum + 1e-8)
    wb = wb / (weight_sum + 1e-8)
    wc = wc / (weight_sum + 1e-8)
    wd = wd / (weight_sum + 1e-8)

    # Compute the interpolated value
    result = wa * Ia + wb * Ib + wc  * Ic + wd  * Id
#     print("result",result)
    # result[weight_sum == 0] = float('nan')  # If all weights are zero, output NaN

    return result

def quadratic_spline_shared_knots(x_knots, y_knots, x_query):
    """
    Vectorized quadratic spline interpolation with shared knots, batched queries.
    
    x_knots: (N,)  - shared x spline knots (must be sorted)
    y_knots: (N,)  - shared y spline knots
    x_query: (B, M) - batched query x-values
    
    Returns:
        y_query: (B, M) - interpolated outputs
    """
    B, M = x_query.shape
    N = x_knots.shape[0]

    # Precompute h and derivatives
    h = x_knots[1:] - x_knots[:-1]  # (N-1,)
    dy = (y_knots[2:] - y_knots[:-2]) / (x_knots[2:] - x_knots[:-2])  # (N-2,)
    dy = torch.cat([dy[:1], dy, dy[-1:]])  # (N,)

    # Quadratic spline coefficients
    a = (dy[1:] + dy[:-1] - 2 * (y_knots[1:] - y_knots[:-1]) / h) / h  # (N-1,)
    b = (3 * (y_knots[1:] - y_knots[:-1]) / h - 2 * dy[:-1] - dy[1:])  # (N-1,)
    c = y_knots[:-1]  # (N-1,)

    # Bin index for each query
    idx = torch.searchsorted(x_knots, x_query, right=False)  # (B, M)
    idx = torch.clamp(idx - 1, 0, N - 2)

    # Gather coeffs for query points
    a_sel = a[idx]  # (B, M)
    b_sel = b[idx]
    c_sel = c[idx]
    x0_sel = x_knots[idx]

    dx = x_query - x0_sel  # (B, M)
    y_query = a_sel * dx**2 + b_sel * dx + c_sel  # (B, M)

    return y_query

