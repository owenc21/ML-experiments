""" 
This is an implementation of a standard feed-forward multi-layer perceptron neural network.
Code is based off of Michael Nielsen's implementation of network2.py
(https://github.com/mnielsen/neural-networks-and-deep-learning)

This architecture was tested on the MNIST dataset (using loaders for the data provided by
Michael Nielsen's repo)

This implementation employs a cross-entropy cost function, a weight initializer that uses Gaussian
random variables with mean 0 and standard deviation 1/sqrt(n_in), where n_in is the number of input
weights into the neuron, L2 regulariazation, and a backpropagation algorithm written from scratch
"""
import numpy as np


class Network():

    """
    Constructor for network
    @param sizes    List of sizes for each respective layer in the network
    """
    def __init__(self, sizes):
        self.sizes = sizes
        """
        Initialize weights and biases using Gaussian random variables
        Mean of 0, std of (1 for biases, 1/sqrt(n_in) for weights)
        """
        self.biases = [np.random.randn(1,x) for x in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for
                        (x,y) in zip(sizes[1:], sizes[:-1])]
    
    """
    Performs a forward pass of the given input argument, returns the output
    of network
    @param input    The input to perfom forward pass on
    @returns        Output of network
    """
    def feedforward(self, input):
        #Iteratively pass the input through lists of weights and biases
        for weight, bias in zip(self.weights, self.biases):
            input = self.activationF(np.dot(weight, input), + bias)
        
        return input
    
    """
    Backpropagates (calculates the partial derivaties of weights and biases) with a single input argument,
    using the given expected output
    @param input    The input to the network to perform back propagation on
    @param expected The expected network output with input
    @returns 2-tuple of partial derivaites of all biases and weights, respectively
    """
    def backpropagation(self, input, expected):
        # Define gradient matrices/lists for weights and biases (respectively) to store the graident of cost
        # with respect to each weight/bias
        partial_w = [np.zeros(w.shape) for w in self.weights]
        partial_b = [np.zeros(b.shape) for b in self.biases]

        # Perform a forward pass of the input and store the activations as well as pre-sigmoid activation values
        activation = input
        activations = [input]
        ps_activations = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            activation = self.activationF(z)
            ps_activations.append(z)
            activations.append(activations)
        
        # Perform the first backwards pass
        delta = self.output_delta(activations[-1], expected)
        partial_b[-1] = delta
        partial_w[-1] = np.dot(delta, activations[-2].transpose())

        # Perform backwards passes iteratively, updating partial derivatives each iteration
        for l in range(2, len(self.sizes)):
            pre_activation = ps_activations[-l]
            deriv = self.activationFP(pre_activation)
            delta = np.dot(self.weights[-l-1].transpose(), delta) * deriv
            partial_b[-l] = delta
            partial_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return(parital_b, partial_w)
        

    """
    Cross-entropy cost function
    Returns the cost of a given output with respect to an expected value
    @param output   The output to calculate the cost for
    @param desired  The desired value of the output
    @returns the cross-entropy cost of output with respect to desired
    """
    def cost(self, output, desired):
        return np.sum(np.nan_to_num(-desired*np.log(output)-(1-desired)*np.log(1-output)))
    
    """
    Calculate the error of the output layer given the output of the layer and the expected value
    @param output   The output of the output layer
    @param desired  Desired output of the output layer
    @returns    Error associated (delta) with the output layer given output and desired
    """
    def output_delta(self, output, desired):
        return output-desired

    """
    Applies activation function to input. Activation function is sigmoid
    @param a    Input to apply function to
    @returns    Result of function with a as input
    """
    def activationF(self, a):
        """Using sigmoid activation function"""
        return 1.0/(1.0+np.exp(-a))
    
    """
    Applies derivative of activation function (sigmoid) to an input
    @param a    Input to apply funciton to
    @returns    Result of derivative evaluation at input
    """
    def activationFP(self, a):
        return self.activationF(a)*(1-self.activationF(a))

