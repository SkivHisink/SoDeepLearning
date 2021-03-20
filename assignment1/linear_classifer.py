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
    if (predictions.ndim == 1):
        Tmp = np.copy(predictions)
        Tmp  -= np.max(predictions) 
        divider = 0;
        for i in range(len(predictions)):
            divider += np.exp(Tmp[i])
        probs = np.exp(predictions)/divider
        return probs
    else:
        Tmp = np.copy(predictions)
        probs = np.copy(predictions)
        max = 0
        for j in range(predictions.shape[0]):
            
            for i in range(predictions.shape[1]):
                if (max<predictions[j][i]):
                    max = predictions[j][i]
            
            for i in range(predictions.shape[1]):
                Tmp[j][i] -= max
            divider = 0.0;
            
            for i in range(predictions.shape[1]):
                Tmp[j][i] = np.exp(Tmp[j][i])
            
            for i in range(predictions.shape[1]):
                divider += Tmp[j][i]
            
            for i in range(predictions.shape[1]):
                probs[j][i] = Tmp[j][i]/divider

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
    loss = reg_strength * np.sum(np.square(W))
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
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    gradient = np.dot(X.T, dprediction)
    
    return loss, gradient


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
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
            loss_sum = 0
            batchesCount = int(num_train / batch_size)
            for batch_i in range(batchesCount):
                loss, grad = linear_softmax(X[batches_indices[batch_i]], self.W, y[batches_indices[batch_i]])
                regLoss, regGrad = l2_regularization(self.W, reg)
                loss += regLoss
                grad += regGrad
                
                self.W -= learning_rate * grad
                loss_sum += loss
                loss_history.append(loss)
            print("Epoch %i, loss: %f" % (epoch, loss))

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
        prediction = np.dot(X, self.W)
        y_pred = np.where(prediction == np.amax(prediction, axis = 1).reshape(np.amax(prediction, 1).shape[0], 1))[1]

        return y_pred



                
                                                          

            

                
