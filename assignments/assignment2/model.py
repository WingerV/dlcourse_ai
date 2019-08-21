import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fully_connected_layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer_1 = ReLULayer()
        self.fully_connected_layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)
        ###self.relu_layer_2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.params()["W1"].grad.fill(0)
        self.params()["B1"].grad.fill(0)
        self.params()["W2"].grad.fill(0)
        self.params()["B2"].grad.fill(0)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        res_fc_1 = self.fully_connected_layer_1.forward(X)
        res_relu_1 = self.relu_layer_1.forward(res_fc_1)
        res_fc_2 = self.fully_connected_layer_2.forward(res_relu_1)
        ###res_relu_2 = self.relu_layer_2.forward(res_fc_2)
        ###loss_softmax, dprediction = softmax_with_cross_entropy(res_relu_2, y)
        ###d_res_relu2 = self.relu_layer_2.backward(dprediction)
        ###d_res_fc_2 = self.fully_connected_layer_2.backward(d_res_relu2)
        loss_softmax, dprediction = softmax_with_cross_entropy(res_fc_2, y)
        d_res_fc_2 = self.fully_connected_layer_2.backward(dprediction)
        
        d_res_relu_1 = self.relu_layer_1.backward(d_res_fc_2)
        d_res_fc_1 = self.fully_connected_layer_1.backward(d_res_relu_1)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        l2_loss_W1, l2_grad_W1 = l2_regularization(self.params()["W1"].value, self.reg)
        l2_loss_B1, l2_grad_B1 = l2_regularization(self.params()["B1"].value, self.reg)
        l2_loss_W2, l2_grad_W2 = l2_regularization(self.params()["W2"].value, self.reg)
        l2_loss_B2, l2_grad_B2 = l2_regularization(self.params()["B2"].value, self.reg)
        self.params()["W1"].grad += l2_grad_W1
        self.params()["B1"].grad += l2_grad_B1
        self.params()["W2"].grad += l2_grad_W2
        self.params()["B2"].grad += l2_grad_B2
        loss = loss_softmax + l2_loss_W1 + l2_loss_B1 + l2_loss_W2 + l2_loss_B2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #print("X ")
        #print(X.shape)
        pred = np.zeros(X.shape[0], np.int)
        #print("W - fully_connected_layer_1")
        #print(self.fully_connected_layer_1.params()["W"].value.shape)
        res_fc_1 = self.fully_connected_layer_1.forward(X)
        #print("res_fc_1 ")
        #print(res_fc_1.shape)
        #print("res_fc_1 = ")
        #print(res_fc_1)
        res_relu_1 = self.relu_layer_1.forward(res_fc_1)
        #print("res_relu_1 ")
        #print(res_relu_1.shape)
        #print("res_relu_1 = ")
        #print(res_relu_1)
        #print("W - fully_connected_layer_2")
        #print(self.fully_connected_layer_2.params()["W"].value.shape)
        res_fc_2 = self.fully_connected_layer_2.forward(res_relu_1)
        #print("res_fc_2 ")
        #print(res_fc_2.shape)
        #print("res_fc_2 = ")
        #print(res_fc_2)
        ###res_relu_2 = self.relu_layer_2.forward(res_fc_2)
        #print("res_relu_2 ")
        #print(res_relu_2.shape)
        #print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq = ", self.params()["W1"].value[0][0])
        #print("res_relu_2 = ")
        #print(res_relu_2)
        ###pred = np.argmax(res_relu_2, axis=1)
        pred = np.argmax(res_fc_2, axis=1)
        #print("pred = ", pred)
        #raise("The end")
        
        return pred
    

    def params(self):
        #print(self.fully_connected_layer_1.params())
        result = {"W1" : self.fully_connected_layer_1.params()["W"], "B1" : self.fully_connected_layer_1.params()["B"], "W2" : self.fully_connected_layer_2.params()["W"], "B2" : self.fully_connected_layer_2.params()["B"]}

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
