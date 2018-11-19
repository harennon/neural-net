import numpy as np
class perceptron(object):
    
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length) * 0.5
        else:
            self.weights = weights

    @staticmethod
    def unit_step_function(x):
        if x > 0.5:
            return 1
        return 0

    def __call__(self, in_data):
        weighted_inputs = self.weights * in_data
        weighted_sum = weighted_inputs.sum()
        return perceptron.unit_step_function(weighted_sum)

    pass

p = perceptron(2)
for x in [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]:
    y = p(np.array(x))
    print(x, y)



