import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = np.sum(np.square(W)) * reg_strength
    grad = 2 * W * reg_strength
    
    return loss, grad


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
    if (predictions.ndim == 1):
        Tmp = np
        Tmp  -= np.max(predictions) 
        divider = 0
        for i in range(len(predictions)):
            divider += np.exp(Tmp[i])
        probs = np.exp(predictions) / divider
        return probs
    else:
        Tmp = np.copy(predictions)
        probs = np.copy(predictions)
        max = 0
        for j in range(predictions.shape[0]):
            for i in range(predictions.shape[1]):
                if (max < predictions[j][i]):
                    max = predictions[j][i] 
            for i in range(predictions.shape[1]):
                Tmp[j][i] -= max
            divider = 0.0
            for i in range(predictions.shape[1]):
                Tmp[j][i] = np.exp(Tmp[j][i])
            for i in range(predictions.shape[1]):
                divider += Tmp[j][i]
            for i in range(predictions.shape[1]):
                probs[j][i] = Tmp[j][i] / divider

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
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_n = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(loss_n) / batch_size
        
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
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    forward_data = np.zeros(1)
    d_result = np.zeros(1)
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #Нормальный(понятный) код
      # tmp = np.zeros((X.shape[0], X.shape[1]))
      # for j in range(0, 2):
      #    for i in range(0, X.shape[j]):
      #         tmp[j][i] = X[j][i] if X[j][i] > 0 else 0
        self.forward_data = X
        # нужный код
        return np.where(X < 0 , 0 , X)

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
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops <- Это относится к
        # бэкварду
        # или и бэкварду и к форварду?!?!
        # В прошлом задании имплементация без циклов работала медленнее
        self.d_result = np.where(self.forward_data < 0 , 0 , 1) * d_out
        return self.d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(X,self.W.value) + self.B.value
        raise Exception("Not implemented!")

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
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        d_input = np.dot(d_out,np.transpose(self.W.value))   
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)
        self.W.grad += np.dot(np.transpose(self.X),d_out)

        #raise Exception("Not implemented!")

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
