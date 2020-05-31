import numpy as np
np.random.seed(42)

class Layer:
    """
    Base class for all the layers in the frame work
    """
    def __init__(self):
        pass

class Dense(Layer):
    """
    Creates a dense layer with the given number of neurons.
    When called, returns the output in the shape (num_neurons, batch_size)
    """
    def __init__(self, units, weight_init='random_normal', use_bias=True, bias_init='zeros'):
        super(Dense, self).__init__()
        self._num_units = units
        self._weight_initializer = weight_init
        self._use_bias = use_bias
        self._bias_initializer = bias_init
        self._weights = None
        self._bias = None

    def __call__(self, input_layer):
        if len(input_layer.shape) == 1:
            input_layer = input_layer.reshape(input_layer.shape[0], -1)
        else:
            input_layer = input_layer.T
        input_shape = np.array(input_layer).shape
        self._weights = 0.10 * np.random.randn(self._num_units, input_shape[0])
        if self._use_bias:
            self._bias = np.zeros((self._num_units, 1), dtype='float32')
        result = np.dot(self._weights, input_layer)
        if self._use_bias:
            result += self._bias
        return result

if __name__ == '__main__':
    inputs = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    res = Dense(3)(inputs)
    print(res)
        

