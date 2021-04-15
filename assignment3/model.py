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
        self.layers = [ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2], filter_size=conv1_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),#2
                       ConvolutionalLayer(in_channels=input_shape[2], out_channels=input_shape[2], filter_size=conv2_channels, padding=2),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, stride=2),#2
                       Flattener(),
                       FullyConnectedLayer(n_input=192, n_output=n_output_classes)]#192

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
        for param in self.params().values():
            param.grad.fill(0.0)
        forward_out = X
        for layer in self.layers:
            forward_out = layer.forward(forward_out)    
        loss, d_out = softmax_with_cross_entropy(forward_out, y)
        backward_out = d_out
        for layer in reversed(self.layers):
            backward_out = layer.backward(backward_out)
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        forward_out = X
        for layer in self.layers:
            forward_out = layer.forward(forward_out)
        y_pred = np.argmax(forward_out, axis = 1)
        return y_pred

    def params(self):
        result = {  
            'FLM': self.layers[0].params()['W'], #First Layer Weight
            'FLB': self.layers[0].params()['B'], #First Layer Bias
            'SLW': self.layers[3].params()['W'], #Second Layer Weight
            'SLB': self.layers[3].params()['B'], #Second Layer Bias
            'TLW': self.layers[7].params()['W'], #Third Layer Weight
            'TLB': self.layers[7].params()['B']  #Third Layer Bias
            }
        return result
