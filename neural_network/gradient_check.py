# gradient_check.py

import numpy as np
from neural_network import initialize_parameters, model_forward_with_dropout, model_backward_with_regularization
from neural_network import compute_cost_with_regularization


def dictionary_to_vector(parameters, debug=False):

    theta = []

    L = len(parameters) // 2

    for l in range(1, L+1):

        if (debug == True):
            print('Shape of W before reshape: ', np.shape(parameters['W' + str(l)]))
            print('Shape of b before reshape: ', np.shape(parameters['b' + str(l)]))

        W = parameters['W' + str(l)].reshape(-1, 1)
        b = parameters['b' + str(l)].reshape(-1, 1)

        if (debug == True):
            print('\t Shape of W after reshape: ', np.shape(W))
            print('\t Shape of b after reshape: ',  np.shape(b))

        theta.append(W)
        theta.append(b)

    theta_vec = np.concatenate(theta, axis = 0)

    return theta_vec
 
def vector_to_dictionary(theta_vec, layers_dims):
    
    L = len(layers_dims)

    parameters = {}
    
    current_index = 0

    for l in range(1, L):
        w_shape = (layers_dims[l], layers_dims[l-1])
        w_size = w_shape[0] * w_shape[1]
        b_shape = (layers_dims[l], 1)
        b_size = b_shape[0] * b_shape[1]

        parameters['W' + str(l)] = theta_vec[current_index: current_index + w_size].reshape(w_shape)
        current_index = current_index + w_size
        parameters['b' + str(l)] = theta_vec[current_index: current_index + b_size].reshape(b_shape)
        current_index = current_index + b_size

    return parameters 

def gradients_to_vector(grads, parameters):

    grad_vector = []
    L = len(parameters)//2
    for l in range(1, L+1):

        dW = grads['dW' + str(l)].reshape(-1, 1)
        db = grads['db' + str(l)].reshape(-1, 1)

        grad_vector.append(dW)
        grad_vector.append(db)

    grad_vector = np.concatenate(grad_vector, axis=0)

    return grad_vector 

def forward_prop_cost(X, Y, parameters, lambd = 0):
    # model_forward_with_dropout will not make use of keep_prob with dropout = False
    # and will act as normal model_forward
    AL, caches, dropout_masks = model_forward_with_dropout(X, parameters, dropout=False, keep_prob=1)
    cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
    # model_backward_with_regularization will also not implement dropout but still requires
    # dropout_masks, which will be the output from model_forward_with_dropout which returns
    # an empty dictionary. 

    return cost


def gradient_check(parameters, grads, X, Y, layers_dims, epsilon=1e-7, print_msg = True):
    theta = dictionary_to_vector(parameters)
    grad = gradients_to_vector(grads, parameters)

    num_parameters = theta.shape[0]
    grad_approx = np.zeros((num_parameters, 1))

    for params_i in range(num_parameters):

        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)

        theta_plus[params_i] += epsilon
        theta_minus[params_i] -= epsilon

        # we compute the cost after changing each individual parameter. 
        J_plus = forward_prop_cost(X, Y, vector_to_dictionary(theta_plus, layers_dims))
        J_minus = forward_prop_cost(X, Y, vector_to_dictionary(theta_minus, layers_dims))

        grad_approx[params_i] = (J_plus - J_minus) / (2.0 * epsilon)

    numerator = np.linalg.norm(grad - grad_approx)

    # add 1e-8 term for stability/prevent divide by 0
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx) + 1e-8
    difference = numerator/denominator

    if print_msg:
        print("Approx vs. Backprop difference = " + str(difference))

    return difference

np.random.seed(10)

X_subset = np.random.randn(4, 50)
Y_subset = (np.random.randn(1, 50) > 0.5).astype(int)

print(X_subset, X_subset.shape)
print(Y_subset, Y_subset.shape)


layers_dims = [X_subset.shape[0], 20, 7, 5, 3, 1]

parameters = initialize_parameters(layers_dims)

#for this test case dropout is disabled
AL, caches, dropout_masks = model_forward_with_dropout(X_subset, parameters)
grads = model_backward_with_regularization(AL, Y_subset, caches, parameters, lambd = 0, 
            dropout=False, dropout_masks=dropout_masks, keep_prob=1)

difference = gradient_check(parameters, grads, X_subset, Y_subset, layers_dims, epsilon=1e-5)

if difference > 1e-2:
    print("There might be a mistake in the backward propagation. difference = ", difference)
elif difference < 1e-7:
    print("backward propagation works perfectly! difference = ", difference)
else:
    print("backward propagation is close enough. difference = ", difference)