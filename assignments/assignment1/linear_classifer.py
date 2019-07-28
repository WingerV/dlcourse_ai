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
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    probs = predictions.copy()
    probs -= np.max(probs, axis=1)[:, np.newaxis]
    #probs -= np.max(probs)
    #print("probs = ", probs)
    exp_sum = np.sum(np.exp(probs), axis=1)
    #print("sum1 = ", np.sum(np.exp(probs), axis=1, keepdims=True))
    #print("sum2 = ", np.outer(exp_sum, np.ones(probs.shape[1])))
    probs = np.exp(probs)/np.outer(exp_sum, np.ones(probs.shape[1]))
    #probs = np.exp(probs)/np.sum(np.exp(probs), axis=1, keepdims=True)
    #print("probs2 = ", probs)
    #print("Prediction = ", predictions)
    return probs


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
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    #res = - np.log(probs[np.arange(len(probs)), target_index])
    #print(res)
    target_index = target_index.flatten()
    str_index_arr = np.arange(target_index.shape[0])
    #loss = -np.sum(np.log(probs[(str_index_arr, target_index)]))/target_index.shape[0]
    loss = - np.log(probs[(str_index_arr, target_index)]).mean()
    #print("loss = ", loss)
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    #loss = -np.sum(np.log(predictions[target_index]))
    
    #dprediction[target_index] -= loss
    #dprediction = dprediction-loss
    softmax_arr = softmax(predictions)
    target_index = target_index.flatten()
    loss = cross_entropy_loss(softmax_arr, target_index)
    mask = np.zeros_like(predictions)
    str_index_arr = np.arange(target_index.shape[0])
    mask[(str_index_arr, target_index)] = 1
    dprediction = (softmax_arr.copy() - mask)/target_index.shape[0]
    
    return loss, dprediction


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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpredictions)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")
            loss = 0
            for batch_ind in batches_indices:
                loss_sm, grad_sm = linear_softmax(X[batch_ind], self.W, y[batch_ind])
                loss_reg, grad_reg = l2_regularization(self.W, reg)
                loss += loss_sm + loss_reg
                self.W -= learning_rate * (grad_sm + grad_reg)
                #print("grad_sm = \n", grad_sm)
                #print("grad_reg = \n", grad_reg)
                #print("W = \n", self.W)
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred



                
                                                          

            

                
