import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.layers = {"conv1": ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
                  "relu1": ReLULayer(),
                  "maxpool1": MaxPoolingLayer(4, 4),
                  "conv2": ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
                  "relu2": ReLULayer(),
                  "maxpool2": MaxPoolingLayer(4, 4),
                  "flatten": Flattener(),
                  "fc": FullyConnectedLayer((input_shape[0]//16)*(input_shape[1]//16)*conv2_channels, n_output_classes)}
                  

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        #raise Exception("Not implemented!")
        for param in self.params().values():
            param.grad.fill(0)
        internal_X = X
        for layer in self.layers.values():
            internal_X = layer.forward(internal_X)
        loss, d_out = softmax_with_cross_entropy(internal_X, y)
        for layer in reversed(list(self.layers.values())):
            d_out = layer.backward(d_out)
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        #raise Exception("Not implemented!")
        internal_X = X
        for layer in self.layers.values():
            internal_X = layer.forward(internal_X)
        return np.argmax(internal_X, axis=1)

    def params(self):
        result = {"{}_{}".format(layer_name, param_name): param_value
                  for layer_name, layer_value in self.layers.items()
                  for param_name, param_value in layer_value.params().items()                
                 }

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")

        return result
