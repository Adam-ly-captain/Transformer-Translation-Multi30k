import torch


def get_criterion(criterion_name):
    """Returns the criterion function based on the given name."""
    criterions = {
        'cross_entropy': cross_entropy_criterion,
        'mse': mse_criterion,
        'mae': mae_criterion,
    }

    criterion_func = criterions.get(criterion_name.lower())
    if not criterion_func:
        raise ValueError(f"Criterion '{criterion_name}' not recognized. Available options: {list(criterions.keys())}")

    return criterion_func


def cross_entropy_criterion(output, target, ignore_index=-100):
    """
    Computes the cross-entropy loss between the output and target tensors.
    
    Args:
        output (torch.Tensor): The model output of shape (batch_size, seq_length, vocab_size).
        target (torch.Tensor): The target tensor of shape (batch_size, seq_length).

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    # Reshape the output and target tensors
    output = output.view(-1, output.size(-1))
    target = target.view(-1)

    # Compute the cross-entropy loss
    loss = torch.nn.functional.cross_entropy(output, target, ignore_index=ignore_index)

    return loss


def mse_criterion(output, target):
    """
    Computes the mean squared error (MSE) loss between the output and target tensors.

    Args:
        output (torch.Tensor): The model output of shape (batch_size, seq_length, vocab_size).
        target (torch.Tensor): The target tensor of shape (batch_size, seq_length, vocab_size).

    Returns:
        torch.Tensor: The computed MSE loss.
    """
    # Reshape the output and target tensors
    output = output.view(-1, output.size(-1))
    target = target.view(-1, target.size(-1))

    # Compute the MSE loss
    loss = torch.nn.functional.mse_loss(output, target)

    return loss


def mae_criterion(output, target):
    """
    Computes the mean absolute error (MAE) loss between the output and target tensors.

    Args:
        output (torch.Tensor): The model output of shape (batch_size, seq_length, vocab_size).
        target (torch.Tensor): The target tensor of shape (batch_size, seq_length, vocab_size).

    Returns:
        torch.Tensor: The computed MAE loss.
    """
    # Reshape the output and target tensors
    output = output.view(-1, output.size(-1))
    target = target.view(-1, target.size(-1))

    # Compute the MAE loss
    loss = torch.nn.functional.l1_loss(output, target)

    return loss