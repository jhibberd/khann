import math
import random


LEARNING_RATE = 0.005

class Node(object):
    """The most general expression of a node. It simply contains an output 
    value. Input nodes are of this type.
    """

    def __init__(self):
        self.output = None


class HiddenOrOutputNode(Node):
    """A specialised type of node that contains weights/connections to parent
    nodes and the ability to calculate it's output. Hidden and output nodes
    are of this type.
    """

    def __init__(self, parent_nodes):
        """Define parent nodes that this node connects to (upstream).

        Args:
            parent_nodes: ([Node]) List of parent nodes.
        """
        super(HiddenOrOutputNode, self).__init__()
        self.error_term = None
        self._parent_nodes = parent_nodes
        self._init_weights()

    def calc_output(self):
        """Calculate the node's output given its weights and the output values
        from parent nodes.
        """

        # Collect the outputs from the parent nodes as a vector.
        outputs = map(lambda n: n.output, self._parent_nodes)

        # Get the sum of the outputs multiplied by the weights.
        o = 0
        for w,x in zip(self.weights, outputs):
            o += w * x

        # Pass the output through the sigmoid function.
        o = self._sigmoid(o)

        self.output = o

    def _init_weights(self):
        """Initialise weights to random floating point values 0 < n < 1."""
        self.weights = []
        for _ in self._parent_nodes:
            r = random.random()
            self.weights.append(r)

    @staticmethod
    def _sigmoid(x):
        """Sigmoid squashing function to allow for solution complexity to go
        beyond linear functions.
        """
        return 1 / (1 + math.exp(-x))

    def update_weights(self):
        for i in range(len(self.weights)):
            delta = LEARNING_RATE * self.error_term * self._parent_nodes[i].output
            self.weights[i] += delta


class OutputNode(HiddenOrOutputNode):
    
    # p.98
    def calc_error_term(self, target_output):
        error = float(target_output) - self.output
        self.error_term = self.output * (1 - self.output) * error # This is just the delta


class HiddenNode(HiddenOrOutputNode):

    def calc_error_term(self, weights_from_output_nodes, output_nodes):
        error = 0.0
        # TODO Wrong! Weights should be those between output_nodes and this 
        # node, not between this node and its parents.
        for w,on in zip(weights_from_output_nodes, output_nodes):
            error += w * on.error_term
        self.error_term = self.output * (1 - self.output) * error # This is just the delta


class Network(object):

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        """Construct an ANN containing a specific number of input, hidden and
        output nodes.
        """

        # Need pointers to the following sets of nodes.
        self._input_nodes = []
        self._output_nodes = []
        self._hidden_nodes = []
        self._hidden_or_output_nodes = []

        # Construct input nodes.
        for i in range(num_input_nodes):
            n = Node()
            self._input_nodes.append(n)

        # Construct hidden nodes.
        for i in range(num_hidden_nodes):
            n = HiddenNode(self._input_nodes)
            self._hidden_nodes.append(n)
    
        # Construct output nodes.
        for i in range(num_output_nodes):
            n = OutputNode(self._hidden_nodes)
            self._output_nodes.append(n)

        self._hidden_or_output_nodes = self._hidden_nodes + self._output_nodes

    def evaluate(self, vector, target_outputs):
        """Given an input vector, generate the corresponding output vector
        the current weights.
        """
       
        # Load the vector into the input nodes.
        for node, x in zip(self._input_nodes, vector):
            node.output = x

        # Starting with the first hidden node and moved "forward" through the
        # network, request that each node calculate its output (given its
        # current weights).
        map(lambda n: n.calc_output(), self._hidden_or_output_nodes)

        # Collect the output values of all output nodes as a vector.
        outputs = map(lambda n: n.output, self._output_nodes)

        # TODO Perhaps calculate total error here and only continue if total
        # error is above a threshold (or hasn't yet converged).
        e = error(outputs, target_outputs)
        print "%s = %s -> %s (er %s)" % (vector, target_outputs, outputs, e)
        if e == 0:
            raise Exception("Converged")

        # Calculate the error term for all output nodes.
        for t, n in zip(target_outputs, self._output_nodes):
            n.calc_error_term(t)

        # Calculate the error term for all hidden nodes.
        for i,n in enumerate(self._hidden_nodes):

            # Get output node weights pointing to that hidden node.
            weights = map(lambda n: n.weights[i], self._output_nodes)

            n.calc_error_term(weights, self._output_nodes)

        # Update all weights accordingly.
        for n in self._hidden_or_output_nodes:
            n.update_weights()


def error(actual_output, target_outout):
    e = 0
    for actual, target in zip(actual_output, target_output):
        e += math.pow(target - actual, 2)
    return e / 2

# Attempt to learn XOR function
# TODO Then try binary addition
training_set = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0]),
    ]
nw = Network(num_input_nodes=2, num_hidden_nodes=2, num_output_nodes=1)

import time
from itertools import cycle

for input_vector, target_output in cycle(training_set):
    nw.evaluate(input_vector, target_output)
    #time.sleep(.5)

# TODO After training the weights, run the input data over the ANN again and 
# check classifications are as expected.

