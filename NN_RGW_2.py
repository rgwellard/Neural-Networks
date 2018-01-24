""" Basic Neural Network package based on Michael Neilsen online book "Neural Networks and Deep Learning"

    Method of use: use external loader to pre-process input data and instantiate NN with NN parameters, launch NN by calling SGD with hyper-parameters

    Definitions

    neural network (NN)     - machine learning construct loosely based on biological neural systems, NN
    neuron                  - individual decision elements in a NN
    layers                  - NN are organized into layers of neurons consisting of input, hidden and output layers
                            - input layers take in data and feed it forward to the hidden layers of the NN
                            - hidden layers process the input and determine "features" which are combined to detect patterns in tha data    
                            - output layers provide the "decision" of the NN, value of the output is the confidence of the NN in making a valid match 
    weights                 - Wi, number applied to output from a neuron i in layer j to feed into a neuron in the next layer
    bias                    - Bi, number applied to each neuron to set sensitivity of neuron to "activation"
    neuron output           - Oi, value of an individual neuron, f(SUM(Wi-1 * Oi-1) - Bi)
    activation function     - function to smooth outputs at each neuron so as small deltas to weights & biases result in small changes to outputs (practical requirement for learning)
    
    NN parameters           - parameters defining construction of NN; number of layers, neurons in each layer, weights, biases
    NN hyper-parameters     - parameters controlling how NN is trained; number of mini-batches, mini-batch size, learning rate                
    mini-batch              - statistically valid sample of training data used to determine incremental learning adjustments
    mini-batch size         - number of samples in mini-batch
    learning rate           - small constant used to adjust trade-off of NN learning rate vs risk of over-shooting minimum
    training data           - data used to train the network
    validation data         - data used to adjust NN hyper-parameters to optimum values
    test data               - data used to test performance of NN
    
Based on the source by Michael Nielsen, "Neural Networks and Deep Learning" and accompanied by the following license

MIT License

Copyright (c) 2012-2015 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import random
import numpy as np          

class Network(object):                      # define class of neural network
    """
    Initialization route for NN class, establishes number of layers and neurons in each layer
    
    self.biases         - layer-by-layer list of arrays of dimension #of neurons x 1 for each layer in NN
                          starts at 2nd layer (index "1"), goes to last, eg. output layer (index ":")
                          arrays loaded with initialized biases (mean = 0.0, sigma = 1.0)
    self.weights        - layer-by-layer list of arrays of dimension #of neurons in each layer x #of neurons in next layer of NN
                          starts at 1st layer, eg. input layer (index ":"), goes to 2nd last (index "-1")
                          arrays loaded with initialized weights (mean = 0.0, sigma = 1.0)                          
    """

    def __init__(self, sizes):              # initialize neural network with string defining layers and respective neurons in network
       
        self.num_layers = len(sizes)        # initialize number of layers in network
        self.sizes = sizes                  # internal variable representing network topology
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]                        # list comprehension
                                                                                        
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]   # list comprehension
        
#       print(list(zip(sizes[:-1], sizes[1:])))

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ 
        Stochastic Gradient Descent Algorithm, SGD
        
        training_data   - data the NN is trained on, list of tuples representing input data and correct result
        epochs          - conventional name for one "training cycle" of the NN when using SGD
        mini_batch_size - number of training record the NN will be trained on in one epoch, eg. not all records are used
        eta             - learning rate, too small and the NN will be slow in converging, too large and minimum could be overshot
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            print("Using test data to determine NN performance after each mini-batch")
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):             # for each of the number of specified epochs
            random.shuffle(training_data)   # build mini-batch by shuffling training data in place
                                            # and selecting number of samples equal to min-batch size
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                                            
                                            # perform one iteration of gradient descent algorithm using mini-batch and apply to NN
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)
                
                                            # Output training progress
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
                               
    def evaluate(self, test_data):
        """
        Returns the number of test inputs for which the NN outputs the correct result
        Note that the NN's output is assumed to be the index of the final layer neuron having the highest activation, eg. argmax
        """
        test_results = [(np.argmax(self.compute_network(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)       
        
    def compute_network(self, x):           
        """
        Run network against data passed in 'x'
        calculate new outputs for each neuron, a' = sigmoid((w * a)+b)
        
        zip method creates an iterator across an array of arrays consisting of biases and weight of the NN
        the first array element is an array of the biases and weights for input to the first hidden layer
        and so on up to and including the output layer
        """

        a = x                               # assign value of neurons in first layer to data
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)     # activation function, sigmoid function applied to neuron weighted inputs
        return a                            # return activations of output layer
       
    def update_mini_batch(self, mini_batch, eta):
        """
        Routine to execute one mini-batch cycle and update network parameters
        calculate changes to weights and biases based on one iteration of gradient descent algorithm
        compute gradient of cost function and move in opposite direction by an amount determined by learning rate
        
        gradC_b         - gradient of cost function (C) wrt biases, gradC_b and gradC_w are layer-by-layer lists of arrays, similar to self.biases and self.weights
        gradC_w         - gradient of cost function (C) wrt weights, gradC_b and gradC_w are layer-by-layer lists of arrays, similar to self.biases and self.weights
        """

        gradC_b = [np.zeros(b.shape) for b in self.biases]                              # list comprehension, shape = shape of biases for each layer 
        gradC_w = [np.zeros(w.shape) for w in self.weights]                             # list comprehension, shape = shape of weights array for each layer
        
        for x, y in mini_batch:

# Sum across all weights and biases to create gradient of cost function        
            delta_gradC_b, delta_gradC_w = self.backprop(x, y)
            gradC_b = [gC_b + delta_gC_b for gC_b, delta_gC_b in zip(gradC_b, delta_gradC_b)]
            gradC_w = [gC_w + delta_gC_w for gC_w, delta_gC_w in zip(gradC_w, delta_gradC_w)]
            
# Make adjustments to weights and biases for current mini-batch          
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, gradC_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, gradC_b)]

    def backprop(self, x, y):
        """
        Return a tuple (gradC_b, gradC_w) representing gradient for the cost function C_x
        """
        gradC_b = [np.zeros(b.shape) for b in self.biases]
        gradC_w = [np.zeros(w.shape) for w in self.weights]
        
# feedforward pass
        activation = x
        activations = [x]                   # list to store all the activations, layer by layer
        zs = []                             # list to store all the z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
        
            z = np.dot(w, activation)+b     # calculate weighted cost
            zs.append(z)                    # save weighted input calculated in feedforward pass for use in backwards pass
            activation = sigmoid(z)         # calculate activation from weighted cost apply sigmoid function
            activations.append(activation)  # save activations calculated in feedforward pass for use in backwards pass
            
# backpropagate pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        gradC_b[-1] = delta
        gradC_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers): # Note: for loop uses negative indexes for weights and activations to implement BP starting at output layer
        
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            gradC_b[-l] = delta
            gradC_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (gradC_b, gradC_w)           # return gradients of cost function

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives, partial C_x / partial a for the output activations
        """
        return (output_activations-y)

#### Miscellaneous functions

def sigmoid(z):                             
    """
    sigmoid function definition, smooths out "raw" outputs from each neuron and normalizes
    output between 0 and 1
    ensures that small changes in weights and biases only result in small changes to outputs
    permitting practical learning implementations    
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    derivative of sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
