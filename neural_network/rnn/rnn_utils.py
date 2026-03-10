import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_adam(parameters):
    """
    Initializes the Adam optimizer state dictionaries v and s.

    Arguments:
    parameters -- python dictionary containing model parameters:
                  parameters["W" + str(l)] = Wl
                  parameters["b" + str(l)] = bl

    Returns:
    v -- dictionary containing first-moment estimates:
         v["dW" + str(l)] and v["db" + str(l)]
    s -- dictionary containing second-moment estimates:
         s["dW" + str(l)] and s["db" + str(l)]
    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(
    parameters,
    grads,
    v,
    s,
    t,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    Update parameters using the Adam optimization algorithm.

    Arguments:
    parameters -- dictionary containing parameters:
                  parameters['W' + str(l)], parameters['b' + str(l)]
    grads      -- dictionary containing gradients:
                  grads['dW' + str(l)], grads['db' + str(l)]
    v          -- dictionary with first-moment estimates
    s          -- dictionary with second-moment estimates
    t          -- current optimization step (time step, starting from 1)
    learning_rate -- scalar learning rate
    beta1 -- exponential decay rate for first-moment estimates
    beta2 -- exponential decay rate for second-moment estimates
    epsilon -- small scalar to prevent division by zero

    Returns:
    parameters -- updated parameters
    v          -- updated first-moment estimates
    s          -- updated second-moment estimates
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads[
            "dW" + str(l + 1)
        ]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads[
            "db" + str(l + 1)
        ]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1**t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1**t)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (
            grads["dW" + str(l + 1)] ** 2
        )
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (
            grads["db" + str(l + 1)] ** 2
        )

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2**t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2**t)

        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)]
            - learning_rate
            * v_corrected["dW" + str(l + 1)]
            / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)]
            - learning_rate
            * v_corrected["db" + str(l + 1)]
            / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)
        )

    return parameters, v, s
