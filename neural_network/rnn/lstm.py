from rnn_utils import sigmoid, tanh, softmax
import numpy as np

def lstm_cell_forward(xt, h_prev, c_prev, parameters):
    """
    Single forward step of an LSTM cell with peephole connections.

    Arguments:
    xt      -- input at time t, shape (n_x, m)
    h_prev  -- previous hidden state, shape (n_h, m)
    c_prev  -- previous cell state, shape (n_h, m)
    parameters -- dictionary of LSTM and output-layer parameters

    Returns:
    h_next  -- next hidden state, shape (n_h, m)
    c_next  -- next cell state, shape (n_h, m)
    yt_pred -- prediction at time t, shape (n_y, m)
    cache   -- values needed for backprop
    """
    
    W_fh = parameters["W_fh"] # (n_h, n_h)
    W_fc = parameters["W_fc"]
    W_fx = parameters["W_fx"] # input connections (n_h, n_x)
    b_f = parameters["b_f"]

    W_ih = parameters["W_ih"]
    W_ic = parameters["W_ic"]
    W_ix = parameters["W_ix"] # input connections (n_h, n_x)
    b_i  = parameters["b_i"]

    W_ch = parameters["W_ch"]
    W_cc = parameters["W_cc"]
    W_cx = parameters["W_cx"]   
    b_c  = parameters["b_c"]

    W_oh = parameters["W_oh"]
    W_oc = parameters["W_oc"]
    W_ox = parameters["W_ox"]   
    b_o  = parameters["b_o"]

    W_y = parameters["W_y"]
    b_y = parameters["b_y"]

    #    (n_h, n_h)*(n_h, m)      (n_h, n_h)*(n_h, m)      (n_h, n_X)*(n_x, m)
    z_f = np.dot(W_fc, c_prev) + np.dot(W_fh, h_prev) + np.dot(W_fx, xt) + b_f
    # z_f, z_i, z_c are (n_h, m)

    f   = sigmoid(z_f) # (n_h, m)

    z_i = np.dot(W_ic, c_prev) + np.dot(W_ih, h_prev) + np.dot(W_ix, xt) + b_i
    i   = sigmoid(z_i)

    z_c = np.dot(W_cc, c_prev) + np.dot(W_ch, h_prev) + np.dot(W_cx, xt) + b_c 
    c_i = tanh(z_c)

    c_next = np.add(np.multiply(f, c_prev), np.multiply(i, tanh(z_c)))
    
    z_o = np.dot(W_oc, c_next) + np.dot(W_oh, h_prev) + np.dot(W_ox, xt) + b_o # (n_h, m)

    o = sigmoid(z_o) 

    h_next = np.multiply(o, tanh(c_next)) #(n_H, m)
    
    yt_pred = softmax(np.dot(W_y, h_next) + b_y)

    cache = (
        xt, h_prev, c_prev,
        z_f, z_i, z_c, z_o, 
        f, i, c_i, c_next, o, h_next,
        parameters
    )

    return h_next, c_next, yt_pred, cache

def lstm_forward(x, h0, parameters):
    """
    Forward propagation through an entire LSTM sequence.

    Arguments:
    x          -- input sequence, shape (n_x, m, T_x)
    h0         -- initial hidden state, shape (n_h, m)
    parameters -- dictionary containing LSTM and output-layer parameters

    Returns:
    h          -- hidden states for every time step, shape (n_h, m, T_x)
    y          -- predictions for every time step, shape (n_y, m, T_x)
    c          -- cell states for every time step, shape (n_h, m, T_x)
    caches     -- tuple of (list_of_caches, x)
    """

    caches = []

    n_x, m , T_x = x.shape()
    n_y, n_h = parameters["W_y"].shape

    h = np.zeros((n_h, m, T_x))
    c = np.zeros((n_h, m, T_x))
    y = np.zeros((n_y, m, T_x))

    h_next = h0
    c_next = np.zeros((n_h, m))

    for t in range(T_x):
        xt = x[:, :, t]
        

