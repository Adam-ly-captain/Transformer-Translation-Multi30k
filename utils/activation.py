import torch.nn as nn


def get_activation_module(activation_str):
    activation_str = activation_str.replace("_", "").lower()
    activation_str = activation_str.replace("()", "")
    activation_str = activation_str.strip()
    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "sigmoid":
        return nn.Sigmoid()
    elif activation_str == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_str}")
