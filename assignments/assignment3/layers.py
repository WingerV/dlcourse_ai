import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    return np.mean(-np.log(probs[range(probs.shape[0]), target_index]))

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    return reg_strength * np.sum(W * W), 2 * reg_strength * W


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    probs[range(probs.shape[0]), target_index] -= 1
    return loss, probs / probs.shape[0]


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X_mask = None

    def forward(self, X):
        self.X_mask = (X > 0)
        return X * self.X_mask

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out * self.X_mask

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X

        out_height = height + 2*self.padding - self.filter_size + 1
        out_width = width + 2*self.padding - self.filter_size + 1
        out_arr = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                W_flat = np.reshape(self.W.value, (-1, self.out_channels))
                #print("X: ", max(y-self.padding, 0), min(y+self.filter_size-self.padding, height+1), max(x-self.padding, 0), min(x+self.filter_size-self.padding, width+1))
                
                #X_flat = X[:][max(y-self.padding, 0):min(y+self.filter_size-self.padding, height+1)][max(x-self.padding, 0):min(x+self.filter_size-self.padding, width+1)][:]
                X_flat = X[:, max(y-self.padding, 0):min(y+self.filter_size-self.padding, height+1), max(x-self.padding, 0):min(x+self.filter_size-self.padding, width+1), :]
                #print(X_flat.shape)
                X_flat = np.pad(X_flat, ((0,0), (max(self.padding-y, 0), max(y+self.filter_size-self.padding-height, 0)), (max(self.padding-x, 0), max(x+self.filter_size-self.padding-width, 0)), (0, 0)))
                X_flat = np.reshape(X_flat, (batch_size, -1))
                #print(X_flat.shape)
                #print(W_flat.shape)
                #print(np.dot(X_flat, W_flat))
                out_arr[:, y, x, :] = np.dot(X_flat, W_flat) + self.B.value
        
        return out_arr


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        X_grad = np.zeros(self.X.shape)
        #W_grad = np.zeros(self.W.value.shape)
        #B_grad = np.zeros(self.B.value.shape)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                W_flat = np.reshape(self.W.value, (-1, self.out_channels))
                X_grad_local = np.dot(d_out[:, y, x, :], W_flat.T)
                X_grad_local = np.reshape(X_grad_local, (batch_size, self.filter_size, self.filter_size, self.in_channels))
                cut_list = list(range(self.padding - y))
                cut_list += list(range(height - y + self.padding, self.filter_size))
                if cut_list:
                    #print("cut_list_y = ", cut_list)
                    X_grad_local = np.delete(X_grad_local, cut_list, axis=1)
                cut_list = list(range(self.padding - x))
                cut_list += list(range(width - x + self.padding, self.filter_size))
                if cut_list:
                    #print("cut_list_x = ", cut_list)
                    X_grad_local = np.delete(X_grad_local, cut_list, axis=2)
                #print(X_grad[:, y:y+self.filter_size-self.padding, x:x+self.filter_size-self.padding, :].shape)
                #print(X_grad_local.shape)
                X_grad[:, max(y-self.padding, 0):y+self.filter_size-self.padding, max(x-self.padding, 0):x+self.filter_size-self.padding, :] += X_grad_local
                
                X_flat = self.X[:, max(y-self.padding, 0):min(y+self.filter_size-self.padding, height+1), max(x-self.padding, 0):min(x+self.filter_size-self.padding, width+1), :]
                X_flat = np.pad(X_flat, ((0,0), (max(self.padding-y, 0), max(y+self.filter_size-self.padding-height, 0)), (max(self.padding-x, 0), max(x+self.filter_size-self.padding-width, 0)), (0, 0)))
                X_flat = np.reshape(X_flat, (batch_size, -1))
                
                W_grad_flat = np.dot(X_flat.T, d_out[:, y, x, :])
                self.W.grad += np.reshape(W_grad_flat, self.W.value.shape)
                
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
                

        return X_grad

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X
        out_height = (height-self.pool_size)//self.stride + 1
        out_width = (width-self.pool_size)//self.stride + 1
        out_arr = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                X_local = X[:, self.stride*y:self.stride*y+self.pool_size, self.stride*x:self.stride*x+self.pool_size, :]
                out_arr[:, y, x, :] = np.max(X_local, axis=(1,2))
        return out_arr

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        X_grad = np.zeros(self.X.shape)
        _, out_height, out_width, _ = d_out.shape
        #raise Exception("Not implemented!")
        for y in range(out_height):
            for x in range(out_width):
                X_local = self.X[:, self.stride*y:self.stride*y+self.pool_size, self.stride*x:self.stride*x+self.pool_size, :]
                X_local_flat = np.reshape(X_local, (batch_size, -1, channels))
                max_index_arr = np.argmax(X_local_flat, axis=1)
                for bs_index in range(batch_size):
                    for ch_index in range(channels):
                        y_pool, x_pool = np.unravel_index(max_index_arr[bs_index][ch_index], (X_local.shape[1], X_local.shape[2]))
                        X_grad[bs_index][y_pool+self.stride*y][x_pool+self.stride*x][ch_index] += d_out[bs_index][y][x][ch_index]
        return X_grad               

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        #raise Exception("Not implemented!")
        self.X = X
        return np.reshape(X, (batch_size, -1))

    def backward(self, d_out):
        # TODO: Implement backward pass
        #raise Exception("Not implemented!")
        return np.reshape(d_out, self.X.shape)

    def params(self):
        # No params!
        return {}
