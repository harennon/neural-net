from matplotlib import pyplot as plt
import numpy as np
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

from scipy.stats import truncnorm
def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm( (low - mean) / sd, (upp - mean) / 
                     sd, loc = mean, scale = sd)

class NeuralNetwork:
    
    """
    pre - layers = array of node-count for each layer in the NN
                    len(layers) > 2
                    layers[0] >= 2
          learning_rate = rate used in training
                    learning_rate != 0
    post - initialize local variables, call initialize_net()
    """
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        self.initialize_net()
    
    """
    Initializes the neural network with given input/output and hidden layers.
    For matrix implementation, this means to initialize the weight matrices
    For now, hidden layer will have a steping size of input_size to output_size
    pre - layers[0] > layers[len(layers) - 1]
    post - initialized weight_matrices with random values
    """
    def initialize_net(self):
        self.weight_matrices = []
        #calculate average deviation
        rad = 1 / np.sqrt(self.layers[0])
        #create rv_continous object (random variates)
        W = truncated_normal(mean = 2, sd = 1, low = -rad, upp = rad)
        for i in range(len(self.layers) - 1):
            #use W to generate random value matrices
            self.weight_matrices.append(np.array(W.rvs((self.layers[i + 1], 
                                                       self.layers[i]))))
        print("Initializing weight matrices to random values" ,
              self.weight_matrices)

    """
    Train the neural network with a single input/target_vector pair
    vectors can be tuples, lists, or ndarray
    """
    def train(self, input_vector, target_vector):
        #turning arrays into column vectors
        target_vector = np.array(target_vector).T
        output_vector = self.run(input_vector)
        input_vector = np.array(input_vector, ndmin=2).T

        #calculate error

        pass

    """
    Run the network with an input vector
    input_vector can be a tuple, list, or ndarray
    pre - len(input_vector) == layor[0]
          len(input_vector) >= 2
    post - output_vector calculated from weight_matrices
    """
    def run(self, input_vector):
        print("Running current Neural Network with input : ", input_vector)
        #turning array into column vectors
        output_vector = np.array(input_vector, ndmin=2).T
        for W in self.weight_matrices:
            #matrix multiplying for each weight matrix
            output_vector = np.dot(W, output_vector)
            #passing result through the activation function
            output_vector = activation_function(output_vector)
        return output_vector


if __name__ == "__main__":
    nn = NeuralNetwork(layers = [5, 4, 3], learning_rate = 0.1)
    out = nn.run([1, 2, 3, 4, 5])
    print("Out : ", out)
    