import math
import numpy as np
import random

# TODO Instead of expressing the network as a collection of objects express
# the network as a collection of weights and values/activations(?)

# TODO Perhaps use numpy to speed things up

# TODO Document, commit and copy to try and solve Kaggle problem.

# TODO Handle networks with an arbitrary number of hidden nodes.

LEARNING_RATE = 0.05


class HiddenNode(object):

    def __init__(self):
        self.error_term = None

    def calc_error_term(self, output, weights_from_output_nodes, output_node_error_terms):
        error = 0.0
        for w,oe in zip(weights_from_output_nodes, output_node_error_terms):
            error += w * oe
        self.error_term = output * (1 - output) * error

# TODO Reference book and use correct terminology
# TODO Rename to ANN
class Network(object):

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        """Construct an ANN containing a specific number of input, hidden and
        output nodes.
        """

        # Number of nodes in input, hidden and output layers.
        topology = [2, 2, 1] 

        # TODO rename to self._o
        # Define arrays to hold output values for each node.
        self._x = map(lambda n: np.zeros(n), topology)

        # Define arrays to hold error terms.
        self._e = map(lambda n: np.zeros(n), topology)

        # Initialise weights in generalised algorithm that accounts for 
        # arbitrary network depth.
        self._w = []
        for layer_index, layer_size in enumerate(topology):
            layer = []
            for node_index in range(layer_size):
                if layer_index == 0:
                    node = []
                else:
                    # A weight for each node in the preceding layer
                    node = []
                    for _ in range(topology[layer_index-1]):
                        node.append(random.random())
                layer.append(node)
            self._w.append(layer)
        print self._w

        # Need pointers to the following sets of nodes.
        self._hidden_nodes = []

        # Construct hidden nodes.
        for i in range(num_hidden_nodes):
            n = HiddenNode()
            self._hidden_nodes.append(n)
    
        self._topology = topology

    def evaluate(self, vector):
        """Given an input vector, generate the corresponding output vector
        the current weights.
        """
  
        # Load the vector into the input nodes.
        self._x[0] = np.array(vector)

        # Starting with the first hidden node and moved "forward" through the
        # network, request that each node calculate its output (given its
        # current weights).
        # p.97

        for l in range(1, len(self._topology)):
            for i in range(self._topology[l]):
                o = np.dot(self._w[l][i], self._x[l-1])
                self._x[l][i] = sigmoid(o)

        # Network output vector is outputs from bottom layer.
        return self._x[-1]
    
    def adjust(self, target_outputs):

        # TODO Calculating error terms needs to happen in reverse.
        # Calculate the error term for all output nodes.
        # p.98
        for i,t in enumerate(target_outputs):
            o = self._x[2][i]
            self._e[2][i] = o * (1-o) * (t-o)

        # Calculate the error term for all hidden nodes.
        for i,n in enumerate(self._hidden_nodes):

            # Get output node weights pointing to that hidden node.
            weights = []
            for j in range(self._topology[2]):
                w = self._w[2][j][i]
                weights.append(w)

            n.calc_error_term(self._x[1][i], weights, self._e[2])

        # Update all weights accordingly.

        for i,n in enumerate(self._hidden_nodes):
            for j in range(self._topology[0]):
                delta = LEARNING_RATE * n.error_term * self._x[0][j]
                self._w[1][i][j] += delta

        for i in range(self._topology[2]):
            for j in range(self._topology[1]):
                delta = LEARNING_RATE * self._e[2][i] * self._x[1][j]
                self._w[2][i][j] += delta

    def get_error(self, actual_output, target_output):
        e = 0
        for t,o in zip(target_output, actual_output):
            e += (t-o)**2
        return e * 0.5

def sigmoid(x):
    """Sigmoid squashing function to allow for solution complexity to go
    beyond linear functions.
    """
    return 1 / (1 + math.exp(-x))


# Attempt to learn XOR function
# TODO Then try binary addition
training_set = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0]),
    ]
nw = Network(num_input_nodes=2, num_hidden_nodes=2, num_output_nodes=1)

for i in range(100000):
    error = 0.0
    for input_vector, target_output in training_set:
        actual_output = nw.evaluate(input_vector)
        error += nw.get_error(actual_output, target_output)
        nw.adjust(target_output)
    if i % 100 == 0:
        print "Error %s" % error

for iv, _ in training_set:
    print "%s -> %s" % (iv, nw.evaluate(iv))

