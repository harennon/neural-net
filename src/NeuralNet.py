from matplotlib import pyplot as plt
import run
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
        target_vector = np.array(target_vector, ndmin=2).T
        output_mat = self.run(input_vector)
        output_vector = output_mat[len(self.layers) - 1]
        input_vector = np.array(input_vector, ndmin=2).T
        #calculate error
        print("output vector = ", output_vector)
        print("target vector = ", target_vector)
        #implement backpropagtion
        dE_do = output_vector - target_vector
        self.back_propagation(dE_do, output_mat)
        errors = []
        
        pass


        
    """
        Performs back-propagation on the network given recursively
        dW = matrix of dW for each ***neuron***
        dE_do = partial derivative of error in terms of the sigmoid(o)
        output_mat = matrix of outputs for each layer it went through
        Base case: dW = [], dE_do = (y - t).
                    This happens when we are finding error for the weights
                    leading to the output, or the last matrix
        Recursive case: dW = [some matrix], dE_do = dW_vector from layer after
                    This happens for every intermediate layer + input layer


        for i in reversed(range(len(self.weight_matrices))):
            weight_layer = self.weight_matrices[i]
            #print(weight_layer)
            error_layer = np.zeros(weight_layer.shape)
            #base end case
            if( i == len(self.weight_matrices) - 1 ):
                print(target_vector - output_vector)
                error_layer = (target_vector - output_vector)#not weight_layer, its output_mat[i]
            #recursive inner case
            else:
                #error_layer = np.zeros((len(weight_layer), 1))
                error_layer =  weight_layer * errors[0]
            error_layer *= (output_mat[i+1] * (1.0 - output_mat[i+1]))
            errors.insert(0, error_layer)
            print(errors[0])
            #self.weight_matrices[i] += self.learning_rate * errors[0].dot(output_mat[i])
    """
    def back_propagation(self, dE_do, output_mat, dW = []):
        dW_vector = []

        #current
        output_vector = output_mat[len(output_mat) - 1]
        #sigmoid of current layer
        do_dSum = output_vector * ( 1 - output_vector )
        #neuron value on layer before
        dSum_dw = output_mat[len(output_mat) - 2] 
        #calculate weights for current layer
        dW_vector = np.dot((dE_do * do_dSum), dSum_dw.T) * self.learning_rate
        """
            for o in output_mat[len(output_mat) - 2]:
                dSum_dw = o #neuron value on layer before
                dE_dw = dE_do * do_dSum * dSum_dw
                dW_vector.append(dE_dw)
        """
        dW.insert(0,dW_vector)
        #update weights
        self.weight_matrices[len(output_mat) - 2] += dW[0]
        del output_mat[len(output_mat) - 1]
        
        if(len(output_mat) >= 2):
            #recurse if output_matrix has >= 2 rows --> 1+ weigh    t matrix left
            dW = self.back_propagation(dW_vector, output_mat, dW)
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
        output_mat = [output_vector]
        for W in self.weight_matrices:
            #matrix multiplying for each weight matrix
            output_vector = np.dot(W, output_vector)
            #passing result through the activation function
            output_vector = activation_function(output_vector)
            #add calculated vector to matrix
            output_mat.append(output_vector)
        return output_mat


if __name__ == "__main__":
    nn = NeuralNetwork(layers = [5, 4, 3], learning_rate = 0.1)
    out = nn.train([1, 2, 3, 4, 5], [1, 0, 1])
    print("Out : ", out)
    