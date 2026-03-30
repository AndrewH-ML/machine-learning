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

    c_next = np.add(np.multiply(f, c_prev), np.multiply(i, c_i)) # f * prev + i * update
    
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

    n_x, m , T_x = x.shape
    n_y, n_h = parameters["W_y"].shape

    h = np.zeros((n_h, m, T_x))
    c = np.zeros((n_h, m, T_x))
    y = np.zeros((n_y, m, T_x))

    h_next = h0
    c_next = np.zeros((n_h, m))

    for t in range(T_x):
        xt = x[:, :, t]

        h_next, c_next, y_t, cache  = lstm_cell_forward(
            xt, h_next, c_next, parameters
        )

        h[:, :, t] = h_next
        c[:, :, t] = c_next
        y[:, :, t] = y_t

        caches.append(cache)
    
    caches = (caches, x)

    return h, y, c, caches
        
def lstm_cell_backward(dh_next, dc_o, cache):
    """
    Single backward step of an LSTM cell with peephole connections.

    Arguments:
    dh_next    -- gradient of the loss with respect to the next hidden state h_next,
                  numpy array of shape (n_h, m)
    dc_o       -- gradient of the loss with respect to the updated memory state c_o,
                  numpy array of shape (n_h, m)
    cache      -- values from the forward pass at time step t containing:
                  xt        -- input at time t, shape (n_x, m)
                  h_prev    -- previous hidden state, shape (n_h, m)
                  c_prev    -- previous memory state, shape (n_h, m)
                  z_f       -- forget gate pre-activation, shape (n_h, m)
                  z_i       -- input gate pre-activation, shape (n_h, m)
                  z_c       -- candidate cell pre-activation, shape (n_h, m)
                  z_o       -- output gate pre-activation, shape (n_h, m)
                  f         -- forget gate activation, shape (n_h, m)
                  i         -- input gate activation, shape (n_h, m)
                  c_i       -- candidate cell activation, shape (n_h, m)
                  c_o       -- updated memory state, shape (n_h, m)
                  o         -- output gate activation, shape (n_h, m)
                  h_next    -- next hidden state, shape (n_h, m)
                  parameters -- dictionary containing the peephole LSTM parameters

    Returns:
    gradients  -- python dictionary containing:
                  dxt       -- gradient with respect to input xt, shape (n_x, m)
                  dh_prev   -- gradient with respect to previous hidden state h_prev,
                               shape (n_h, m)
                  dc_prev   -- gradient with respect to previous memory state c_prev,
                               shape (n_h, m)

                  dz_f      -- gradient with respect to z_f, shape (n_h, m)
                  dz_i      -- gradient with respect to z_i, shape (n_h, m)
                  dz_c      -- gradient with respect to z_c, shape (n_h, m)
                  dz_o      -- gradient with respect to z_o, shape (n_h, m)

                  dW_fh     -- gradient with respect to W_fh, shape (n_h, n_h)
                  dW_fc     -- gradient with respect to W_fc, shape (n_h, n_h)
                  dW_fx     -- gradient with respect to W_fx, shape (n_h, n_x)
                  db_f      -- gradient with respect to b_f, shape (n_h, 1)

                  dW_ih     -- gradient with respect to W_ih, shape (n_h, n_h)
                  dW_ic     -- gradient with respect to W_ic, shape (n_h, n_h)
                  dW_ix     -- gradient with respect to W_ix, shape (n_h, n_x)
                  db_i      -- gradient with respect to b_i, shape (n_h, 1)

                  dW_ch     -- gradient with respect to W_ch, shape (n_h, n_h)
                  dW_cc     -- gradient with respect to W_cc, shape (n_h, n_h)
                  dW_cx     -- gradient with respect to W_cx, shape (n_h, n_x)
                  db_c      -- gradient with respect to b_c, shape (n_h, 1)

                  dW_oh     -- gradient with respect to W_oh, shape (n_h, n_h)
                  dW_oc     -- gradient with respect to W_oc, shape (n_h, n_h)
                  dW_ox     -- gradient with respect to W_ox, shape (n_h, n_x)
                  db_o      -- gradient with respect to b_o, shape (n_h, 1)
    """

    # """ cache = (
    #     xt, h_prev, c_prev,
    #     z_f, z_i, z_c, z_o, 
    #     f, i, c_i, c_next, o, h_next,                              
    #     parameters
    # """

    # import input, exposed squashed memory, memory, and activations for
    # a time t. Includes weights
    (xt, h_prev, c_prev, z_f, z_i, z_c, z_o, f, i, c_i, c_o, o, 
        h_next, parameters) = cache
    
    # import  weights
    W_fh = parameters["W_fh"]
    W_fc = parameters["W_fc"]
    W_fx = parameters["W_fx"]

    W_ih = parameters["W_ih"]
    W_ic = parameters["W_ic"]
    W_ix = parameters["W_ix"]

    W_ch = parameters["W_ch"]
    W_cc = parameters["W_cc"]
    W_cx = parameters["W_cx"]

    W_oh = parameters["W_oh"]
    W_oc = parameters["W_oc"]
    W_ox = parameters["W_ox"]

    #get sizing for n_x, n_h, m 
    n_x, m = xt.shape
    n_h, m = h_next.shape 
    tanhc_o = np.tanh(c_o)

    dc_o_tanh = dh_next * o * (1-(tanhc_o)**2)

    #following takes from upstream and from tanh contribution from h_o
    dc_o = dc_o + dc_o_tanh # upstream memory
    
    #backprop through output gate, o = sigmoid(z_o)
    do =  dh_next * tanhc_o # element-wise
    dz_o = do * o*(1-o)

    # following line also contributes to upstream, what the cell sends out...
    dc_o = dc_o + np.dot(W_oc.T, dz_o) #upstream 

    dh_prev = np.dot(W_oh.T, dz_o)
    dxt = np.dot(W_ox.T, dz_o)
    dW_oc = np.dot(dz_o, c_o.T)
    dW_oh = np.dot(dz_o, h_prev.T)
    dW_ox = np.dot(dz_o, xt.T)
    db_o = np.sum(dz_o, axis = 1, keepdims=True)

    #backprop through updated memory
    dc_prev = dc_o * f
    df = dc_o * c_prev
    di = dc_o * c_i
    dc_i = dc_o * i
    dz_c = dc_i * (1- (np.tanh(z_c))**2)

    # backprop through c_i = tanh(z_c)
    dh_prev = dh_prev + np.dot(W_ch.T, dz_c)
    #first contributiion to dc_prev
    dc_prev = dc_prev + np.dot(W_cc.T, dz_c)
    dxt   = dxt + np.dot(W_cx.T, dz_c) 
    
    dW_ch = np.dot(dz_c, h_prev.T)
    dW_cc = np.dot(dz_c, c_prev.T)
    dW_cx = np.dot(dz_c, xt.T)
    db_c  = np.sum(dz_c, axis = 1, keepdims=True)
    
    # backprop through input gate  

    dz_i     = di * (i*(1-i))
    dc_prev  = dc_prev + np.dot(W_ic.T, dz_i)
    dh_prev  = dh_prev +  np.dot(W_ih.T, dz_i)
    dxt       = dxt + np.dot(W_ix.T, dz_i)

    dW_ic = np.dot(dz_i, c_prev.T)
    dW_ih = np.dot(dz_i, h_prev.T)
    dW_ix = np.dot(dz_i, xt.T)
    db_i = np.sum(dz_i, axis= 1, keepdims=True)

    # backprop through forget gate

    dz_f    = df * f*(1-f)
    dc_prev = dc_prev + np.dot(W_fc.T, dz_f)
    dh_prev = dh_prev + np.dot(W_fh.T, dz_f)
    dxt      = dxt + np.dot(W_fx.T, dz_f)

    dW_fc = np.dot(dz_f, c_prev.T)
    dW_fh = np.dot(dz_f, h_prev.T)
    dW_fx = np.dot(dz_f, xt.T)
    db_f  = np.sum(dz_f, axis = 1, keepdims=True)

    gradients = {
        "dxt": dxt,
        "dh_prev": dh_prev,
        "dc_prev": dc_prev,

        "dz_f": dz_f,
        "dz_i": dz_i,
        "dz_c": dz_c,
        "dz_o": dz_o,

        "dW_fh": dW_fh,
        "dW_fc": dW_fc,
        "dW_fx": dW_fx,
        "db_f": db_f,

        "dW_ih": dW_ih,
        "dW_ic": dW_ic,
        "dW_ix": dW_ix,
        "db_i": db_i,

        "dW_ch": dW_ch,
        "dW_cc": dW_cc,
        "dW_cx": dW_cx,
        "db_c": db_c,

        "dW_oh": dW_oh,
        "dW_oc": dW_oc,
        "dW_ox": dW_ox,
        "db_o": db_o,
    }

    return gradients
    
def lstm_backward(dh, caches):
    """
    Backward pass across all timesteps of the LSTM.

    Arguments:
    dh      -- upstream gradient for hidden states, shape (n_h, m, T_x)
    caches  -- (cache_list, x) returned by lstm_forward

    Returns:
    gradients -- dictionary containing input, initial-state, and parameter gradients
    """

    caches, x =  caches
    cache_0 = caches[0]

    (
        xt, h_prev, c_prev,
        z_f, z_i, z_c, z_o,
        f, i, c_i, c_o, o, h_next,
        parameters
    )   = cache_0 

    n_x, m, T_x = x.shape
    n_h, _  = h_next.shape

    #initialize gradients to size of parameters passed in
    dx = np.zeros((n_x, m,  T_x))        
    dh_prev = np.zeros((n_h, m))
    dc_prev = np.zeros((n_h, m))

    dW_fh = np.zeros_like(parameters["W_fh"])
    dW_fc = np.zeros_like(parameters["W_fc"])
    dW_fx = np.zeros_like(parameters["W_fx"])
    db_f = np.zeros_like(parameters["b_f"])

    dW_ih = np.zeros_like(parameters["W_ih"])
    dW_ic = np.zeros_like(parameters["W_ic"])
    dW_ix = np.zeros_like(parameters["W_ix"])
    db_i = np.zeros_like(parameters["b_i"])

    dW_ch = np.zeros_like(parameters["W_ch"])
    dW_cc = np.zeros_like(parameters["W_cc"])
    dW_cx = np.zeros_like(parameters["W_cx"])
    db_c = np.zeros_like(parameters["b_c"])

    dW_oh = np.zeros_like(parameters["W_oh"])
    dW_oc = np.zeros_like(parameters["W_oc"])
    dW_ox = np.zeros_like(parameters["W_ox"])
    db_o = np.zeros_like(parameters["b_o"])

    for t in reversed(range(T_x)):
        grads_t = lstm_cell_backward(
            dh[:,:,t] + dh_prev, 
            dc_prev,
            caches[t]
        )

        dx[:, :, t] = grads_t["dxt"]
        dh_prev = grads_t["dh_prev"]
        dc_prev = grads_t["dc_prev"]
        dW_fh += grads_t["dW_fh"]
        dW_fc += grads_t["dW_fc"]
        dW_fx += grads_t["dW_fx"]
        db_f += grads_t["db_f"]

        dW_ih += grads_t["dW_ih"]
        dW_ic += grads_t["dW_ic"]
        dW_ix += grads_t["dW_ix"]
        db_i += grads_t["db_i"]

        dW_ch += grads_t["dW_ch"]
        dW_cc += grads_t["dW_cc"]
        dW_cx += grads_t["dW_cx"]
        db_c += grads_t["db_c"]

        dW_oh += grads_t["dW_oh"]
        dW_oc += grads_t["dW_oc"]
        dW_ox += grads_t["dW_ox"]
        db_o += grads_t["db_o"]

    dh0 = dh_prev

    gradients = {
        "dx": dx,
        "dh0": dh0,

        "dW_fh": dW_fh,
        "dW_fc": dW_fc,
        "dW_fx": dW_fx,
        "db_f": db_f,

        "dW_ih": dW_ih,
        "dW_ic": dW_ic,
        "dW_ix": dW_ix,
        "db_i": db_i,

        "dW_ch": dW_ch,
        "dW_cc": dW_cc,
        "dW_cx": dW_cx,
        "db_c": db_c,

        "dW_oh": dW_oh,
        "dW_oc": dW_oc,
        "dW_ox": dW_ox,
        "db_o": db_o,
    }

    return gradients