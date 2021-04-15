import numpy as np


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
        Tmp = predictions
        probs = predictions
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        #Нормальный(понятный) код
      # tmp = np.zeros((X.shape[0], X.shape[1]))
      # for j in range(0, 2):
      #    for i in range(0, X.shape[j]):
      #         tmp[j][i] = X[j][i] if X[j][i] > 0 else 0
        self.forward_data = X
        # нужный код
        return np.where(X < 0 , 0 , X)

    def backward(self, d_out):
        self.d_result = np.where(self.forward_data < 0 , 0 , 1) * d_out
        return self.d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        dX = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out)
        self.W.grad += dW
        self.B.grad += dB
        return dX

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
        self.W = Param(np.random.randn(filter_size, filter_size,
                            in_channels, out_channels))

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):#рабоает вроде верно
        batch_size, height, width, channels = X.shape
        out_width = height * width * channels
        out_height = batch_size
        
        self.X = np.zeros((batch_size , height + 2 * self.padding , width + 2 * self.padding , channels))  
        self.X[: , self.padding: self.X.shape[1] - self.padding , self.padding:self.X.shape[2] - self.padding , :] = X
        out_height = self.X.shape[1] - self.filter_size + 1 
        out_width = self.X.shape[2] - self.filter_size + 1       
        result = np.zeros((batch_size , out_height , out_width , self.out_channels))        
        for y in range(out_height):
            for x in range(out_width):
                tmpMatrix = self.X[: , y: y + self.filter_size , x:x + self.filter_size, :]  #
                weightArray = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels,self.out_channels)
                tmp=tmpMatrix.reshape(batch_size , self.filter_size * self.filter_size * self.in_channels) 
                arrayResult = np.dot(tmp, weightArray) + self.B.value          
                matrixResult = arrayResult.reshape(batch_size , 1 , 1, self.out_channels)              
                result[: , y: y + self.filter_size , x:x + self.filter_size, :] = matrixResult
        return result

    def backward(self, d_out): #верно
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros((batch_size, height, width, channels))
        weightArray = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        for x in range(out_width):
            for y in range(out_height):
                X_local_mat = self.X[:, x:x + self.filter_size , y:y + self.filter_size, :]           
                X_arr = X_local_mat.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)
                d_local = d_out[:, x:x + 1, y:y + 1, :]
                dX[:, x:x + self.filter_size , y:y + self.filter_size, :] +=  (np.dot(d_local.reshape(batch_size, -1), weightArray.T)).reshape(X_local_mat.shape)
                self.W.grad += (np.dot(X_arr.T, d_local.reshape(batch_size, -1))).reshape(self.W.value.shape)
                self.B.grad +=  (np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))).reshape(self.B.value.shape)
        return dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]

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
        self.masks = {}

    def forward(self, X): #работает верно
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        output = np.zeros((batch_size, out_height, out_width, channels))
        for x in range(out_width):
            for y in range(out_height):
                self.mask(x = X[:, x * self.stride:x * self.stride + self.pool_size, y * self.stride:y * self.stride + self.pool_size, :], pos=(x, y))
                output[:, x, y, :] = np.max(X[:, x * self.stride:x * self.stride + self.pool_size, y * self.stride:y * self.stride + self.pool_size, :], axis=(1, 2))
        return output

    def backward(self, d_out):#работает верно
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)
        for x in range(out_width):
            for y in range(out_height):
                dX[:, x * self.stride:x * self.stride + self.pool_size, y * self.stride:y * self.stride + self.pool_size, :] += d_out[:, x:x + 1, y:y + 1, :] * self.masks[(x, y)]  
        return dX
#функция взята у одногруппника, поскольку имплементацию без масок было сложно
#понять
    def mask(self, x, pos):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        n_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[n_idx, np.argmax(x, axis=1), c_idx] = 1
        self.masks[pos] = zero_mask

    def params(self):
        return {}

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.Xshape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
       return d_out.reshape(self.Xshape)

    def params(self):
        # No params!
        return {}
