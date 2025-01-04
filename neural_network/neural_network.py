# neural_net.py
import sys
import os
import numpy as np
from optimization_methods import *

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        # dimension of w is (l, l-1)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


# simple forward function to compute z
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache"""
    Z = np.dot(W, A) + b
    # in cache we'll save A, W, b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer 

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";   
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    #linear_cache contains A_prev, W, b used for forward pass
    #activation_cache contains pre-activation value z
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    #caches contains linear & activation cache for all layers
    caches = []
    A = X
    L = len(parameters) // 2

    #range iter below will forward pass w relu up to before the output layer
    for l in range(1, L):
        A_prev = A
        # use relu for forward pass
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 'relu')
        #cache contains both linear_cache and activation_cache
        caches.append(cache)
    #compute forward pass for output layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches


# implement log loss function to create learning metric

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    #get num of training examples
    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    # To make sure cost's shape is whats expected (e.g. this turns [[17]] into 17)
    # np.squeeze ensures we have the correct shape, e.g instead of [[x]], we get X
    cost = np.squeeze(cost)

    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd, l2):
    """
    Compute the cost with or without L2 regularization.

    Arguments:
    AL -- post-activation, output of forward propagation, shape (output size, number of examples)
    Y -- true labels, shape (output size, number of examples)
    parameters -- dictionary containing W and b
    lambd -- regularization hyperparameter
    l2 -- boolean, whether to include L2 regularization

    Returns:
    cost -- regularized or standard cost
    """
    m = Y.shape[1]
    cross_entropy_cost = compute_cost(AL, Y)
    if l2:
        l2_regularization_cost = (lambd / (2 * m)) * sum([np.sum(np.square(parameters['W' + str(l)]))
                                      for l in range (1,len(parameters)//2)])
        cost = cross_entropy_cost + l2_regularization_cost
    else:
        cost = cross_entropy_cost

    return cost


def linear_backward(dZ, cache):
    #get params
    A_prev, W, b = cache
    # get # examples
    m = A_prev.shape[1]

    dW = 1./m * (np.dot(dZ, A_prev.T))
    db = 1./m * (np.sum(dZ, axis = 1, keepdims = True))
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        # compute dZ given inputs
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    #create dictionary to save gradients
    grads = {}
    #number of layers in network
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # make Y same shape as AL

    #initialize backprop
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    #performing backward pass for output layer using sigmoid_backward choice
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')

    #store gradients
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # loop in reverse from l = L-2 to l = 0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def model_backward_with_regularization(AL, Y, caches, parameters, lambd, l2=False):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    refularization -- determines if and what form of regularization to use (l2, dropout)

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    #create dictionary to save gradients
    grads = {}
    #number of layers in network
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # make Y same shape as AL

    #initialize backprop
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    #performing backward pass for output layer using sigmoid_backward choice
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')

    if l2:
        dW_temp +=  (lambd/m) * parameters['W' + str(L)]

    #store gradients
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # loop in reverse from l = L-2 to l = 0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, 'relu')
        if l2:
            dW_temp +=  (lambd/2) * parameters['W' + str(l+1)]
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads
        
    
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    # number of layers, remember params contains both W & b, so we divide by 2
    parameters = params.copy()
    L = len(parameters) // 2

    #start updating!
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]

    return parameters

def predict(X, parameters, y=None):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    # Forward propagation
    probas, caches = model_forward(X, parameters)

    # convert probas to 0/1 predictions
    p = (probas > 0.5).astype(int)

    if(y is not None): 
        
        tp = np.sum((p == 1) & (y == 1))
        tn = np.sum((p == 0) & (y == 0))
        fp = np.sum((p == 1) & (y == 0))
        fn = np.sum((p == 0) & (y == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * ((precision * recall)/(precision + recall)) if (precision + recall) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    return metrics, p



def model_1(X, Y, layers_dims, learning_rate = 0.01, num_iterations=2000, print_cost=True):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # collect costs
    costs = []
    #initialize our params
    parameters = initialize_parameters(layers_dims)
    #perform gradient descent
    for i in range(0, num_iterations):
      #forward prop

      AL, caches = model_forward(X, parameters)
      #compute cost
      cost = compute_cost(AL, Y)
      costs.append(cost)

      #backwardprop
      grads = model_backward(AL, Y, caches)

      #update parameters
      parameters = update_parameters(parameters, grads, learning_rate)

      if ((i!=0) and (print_cost==True) and ((i%100)==0)):
        print("iteration: ", i)
        print("cost: ", costs[i])
      
    return parameters, costs

def model_2_optimized(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs=500, print_cost=True, decay=None, decay_rate=1):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    mini_batch_size -- number of examples desired in each mini batch of examples
    beta -- the beta parameter used for momentum
    beta1, beta2 -- the beta parameters used for adam optimization
    num_epochs -- number of passes through entire batch
    print_cost -- if True, it prints the cost every 100 steps
    decay -- can be used for scheduled or adaptive learning rate decay

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # collect costs
    costs = []
    # intialize counter, t, for Adam update 
    t = 0
    #number of training examples
    m = X.shape[1]
    lr_rates = []
    learning_rate_0 = learning_rate

    #initialize our params
    parameters = initialize_parameters(layers_dims)

    # Initialize optimizer
    if optimizer == "gd":
            pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    seed = 0
        
    for i in range(num_epochs):
        # increment seed to randomize shuffle
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost = 0
      
        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            #forward prop
            AL, caches = model_forward(minibatch_X, parameters)
            #compute cost
            cost += compute_cost(AL, minibatch_Y)
            #backwardprop
            grads = model_backward(AL, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s, t, 
                                                learning_rate, beta1, beta2, epsilon)
        
        average_cost = cost/m

        if decay:
            learning_rate = decay(learning_rate_0, i, decay_rate)

        if ((print_cost==True) and ((i%5)==0)):
            print ("Cost after epoch %i: %f" %(i, average_cost))
            if decay:
                print("learning rate after epoch: %i : %f" % (i, learning_rate))

        if print_cost:
            costs.append(average_cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

def model_3(X, Y, layers_dims, optimizer, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs=500, print_cost=True, decay=None,
            l2_regularization=False, learning_rate = 0.0007, decay_rate = 1, lambd = 0):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    mini_batch_size -- number of examples desired in each mini batch of examples
    beta -- the beta parameter used for momentum
    beta1, beta2 -- the beta parameters used for adam optimization
    num_epochs -- number of passes through entire batch
    print_cost -- prints cost changes if True
    decay -- can be used for scheduled or adaptive learning rate decay

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # collect costs
    costs = []
    # intialize counter, t, for Adam update 
    t = 0
    #number of training examples
    m = X.shape[1]
    lr_rates = []
    learning_rate_0 = learning_rate
    
    #initialize our params
    parameters = initialize_parameters(layers_dims)

    # Initialize optimizer
    if optimizer == "gd":
            pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    seed = 0
        
    for i in range(num_epochs):
        # increment seed to randomize shuffle
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost = 0
      
        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            #forward prop
            AL, caches = model_forward(minibatch_X, parameters)
            #compute cost
            cost += compute_cost_with_regularization(AL, minibatch_Y, parameters=parameters,
                                                      l2=l2_regularization, lambd=lambd)
            #backwardprop
            grads = model_backward_with_regularization(AL, minibatch_Y, caches = caches, parameters=parameters,
                                                        l2=l2_regularization, lambd=lambd)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam": 
                t = t + 1 
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s, t, 
                                                learning_rate, beta1, beta2, epsilon)
        
        average_cost = cost/m

        if decay:
            learning_rate = decay(learning_rate_0, i, decay_rate)

        if ((print_cost==True) and ((i%5)==0)):
            print ("Cost after epoch %i: %f" %(i, average_cost))
            if decay:
                print("learning rate after epoch: %i : %f" % (i, learning_rate))

        
        costs.append(average_cost)

    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('epochs')
    # plt.title("Learning rate = " + str(learning_rate))
    # plt.show()

    return parameters, costs

def search_params(model, X_train, X_test, Y_train, Y_test, params, lr_0s, decay_rates, lambds):
    
    scores = {}

    for lr in lr_0s:
        for decay_rate in decay_rates:
            for lambd in lambds:
                # train model
                # report training and test

                print(f"Evaluating Model with Parameters: lr={lr}, decay={decay_rate}, λ={lambd}")

                parameters, costs = model(X_train, Y_train, *params, learning_rate = lr, decay_rate = decay_rate, 
                                   lambd = lambd)
                
                             # Evaluate metrics

                train_metrics, _ = predict(X_train, parameters, Y_train)
                test_metrics, _ = predict(X_test, parameters, Y_test)
                
                scores[f"lr={lr}_decay={decay_rate}_lambda={lambd}"] = {
                    "train": train_metrics,
                    "test": test_metrics
                }

                plt.plot(costs, label=f"lr={lr}, decay={decay_rate}, λ={lambd}")
                plt.xlabel('Epochs')
                plt.ylabel('Cost')
                plt.title('Cost per Epoch')
                plt.legend(loc='upper right', fontsize='small')
                plt.show()

                print("  Training Metrics:", train_metrics)
                print("  Testing Metrics:", test_metrics)
                print("-" * 50)

    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost per Epoch for Various Hyperparameters')
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

    return scores
 