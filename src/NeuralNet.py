from matplotlib import pyplot as plt
import run
import numpy as np

"""
    vectorized activation function
"""
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

"""
    simple random value generator
"""
from scipy.stats import truncnorm
def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm( (low - mean) / sd, (upp - mean) / 
                     sd, loc = mean, scale = sd)
"""
    Euclidean loss function
"""
def loss(output, target):
    return np.sum((target - output)**2)

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
        print("Initializing weight matrices to random values\n")

    """
    Train the neural network with a single input/target_vector pair
    vectors can be tuples, lists, or ndarray
    """
    def train(self, input_vector, target_vector):
        #turning arrays into column vectors
        target_vector = np.array(target_vector, ndmin=2).T
        output_mat = self.run(input_vector)
        output_vector = output_mat[-1]
        input_vector = np.array(input_vector, ndmin=2).T

        #implement backpropagtion
        dE_do = output_vector - target_vector #[out x 1] vector
        dW = self.back_propagation(dE_do, output_mat, dW = [])

        #update weights
        for i in range(len(dW)):
            self.weight_matrices[i] += dW[i] * (-1) * self.learning_rate
        
        #return loss
        return loss(output_vector, target_vector)
        
    """
        Performs back-propagation on the network given recursively
        dW = matrix of dW for each ***neuron***
        dE_do = partial derivative of error in terms of the sigmoid(o)
        output_mat = matrix of outputs for each layer it went through
        Base case: dW = [], dErr_do = (y - t).
                    This happens when we are finding error for the weights
                    leading to the output, or the last matrix
        Recursive case: dW = [some matrix], dE_do = dW_vector from layer after
                    This happens for every intermediate layer + input layer

    """
    def back_propagation(self, dErr_do, output_mat, i = 1, dW = []):
        dW_mat = []

        #current
        output_vector = output_mat[-(i)]
        #sigmoid of current layer
        do_dnet = output_vector * ( 1 - output_vector )
        #neuron value on layer before
        dnet_dw = output_mat[-(i + 1)] 
        #calculate weights for current layer
        #dE/dw = dE/do o do/dnet x (dnet/dw).T
        dW_mat = (dErr_do * do_dnet).dot(dnet_dw.T)
        #dW_mat should have same dimensions as weight_matrices[i->j]

        #insert calculated dw
        dW.insert(0, dW_mat)

        #recurse if i < len(output_mat) --> 1+ weight matrix left
        if(i + 1 < len(output_mat)):
            #prepare dErr_do for next recurisve call
            #weights from prev layer j to all nodes this layer
            wnetl = np.array(np.sum(self.weight_matrices[-(i)], axis=0), ndmin=2).T
            #print("dErr/do = ", dErr_do.shape, "do_dnet = ", do_dnet.shape, "wnetl = ", wnetl.shape)
            #hadamard multiply dErr/doL o doL/dnetL o wnetL
            dErr_do = np.sum(dErr_do * do_dnet) * wnetl
            dW = self.back_propagation(dErr_do, output_mat, i+1, dW)

        return dW

    """
        trains the network for a fixed number of epochs or until it reaches a loss threshold
        train = [ 
                    [[inputa1, inputa2...],[inputb1, inputb2...],...] ,  
                    [[targeta1,targeta2,...],[targetb1,targetb2,...],...] 
                ]
        if n_epochs = 0, runs until error is below threshold, else run max n_epoch times or
        until error reaches threshold
    """
    def train_network(self, train, n_epochs = 0, threshold=0.1):
        epochs = 0
        print("Starting to train for %d, or when the error is below %.3f\n\n" %(n_epochs, threshold))
        while 1:
            #reset sumError
            sumError = 0

            #train for each training set
            for i in range(len(train[0])):
                sumError += self.train(train[0][i], train[1][i])
            sumError /= (len(train[0]))

            #check if sumError is less than threshold or if n_epochs exceeded
            if (n_epochs != 0 and epochs >= n_epochs) or sumError <= threshold:
                break
            if(epochs % 10 == 0):
                print("Epochs = %d, Error = %.7f" %(epochs, sumError))
            epochs += 1
        print("Finished training\n n_epochs = %d\n sumError = %.5f\n\n" %(epochs, sumError))

    """
    Run the network with an input vector
    input_vector can be a tuple, list, or ndarray
    pre - len(input_vector) == layor[0]
          len(input_vector) >= 2
    post - output_vector calculated from weight_matrices
    """
    def run(self, input_vector):
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


    """
        user-call method for running the Neural net for given input
    """
    def predict(self, input):
        out = self.run(input)
        return out[-1]

'''
if __name__ == "__main__":
    nn = NeuralNetwork(layers = [5, 4, 3], learning_rate = 0.1)
    out = nn.train_network([[[1, 2, 3, 4, 5]], [[1, 0, 1]]], n_epochs=100)
    print("Out : ", out)
'''  