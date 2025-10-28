import torch.optim as optim


def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay=0.0):
    """Returns the optimizer instance based on the given name."""

    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
    }

    optimizer_class = optimizers.get(optimizer_name.lower())
    if not optimizer_class:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized. Available options: {list(optimizers.keys())}")

    return optimizer_class(model_parameters, lr=learning_rate)
