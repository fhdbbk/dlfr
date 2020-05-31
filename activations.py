from layers import Layer
import numpy as np

np.random.seed(42)

class Relu(Layer):
    def __init__(self):
        super(Relu, self).__init__()

    def __call__(self, inp):
        return np.maximum(0, inp)
