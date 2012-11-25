import math
import random


class Node(object):
    """The most general expression of a node. It simply contains an output 
    value. Input nodes are of this type.
    """

    output = None


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


    # TODO(jhibberd) For output node only
    def calc_error_term():
        """
        Compare target with actual output
        Then apply sigmoid derivative.
        """
        pass

    # TODO(jhibberd) For hidden node only
    def calc_error_term():
        """
        Look at *error term* for all downstream nodes
        TODO Each node needs connections upstream and downstream
        Weight each error term by weight of that node (which belongs to output
        node so that could be tricky)
        Then apply sigmoid derivative.
        """
        pass

    def adjust_weights():
        """
        For each node:
            error_term * learning_rate * upstream value for that weight?
        """
        pass


class Network(object):

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):
        """Construct an ANN containing a specific number of input, hidden and
        output nodes.
        """

        # Need pointers to the following sets of nodes.
        self._input_nodes = []
        self._output_nodes = []
        self._hidden_or_output_nodes = []

        # Construct input nodes.
        for i in range(num_input_nodes):
            n = Node()
            self._input_nodes.append(n)

        # Construct hidden nodes.
        hidden_nodes = []
        for i in range(num_hidden_nodes):
            n = HiddenOrOutputNode(self._input_nodes)
            hidden_nodes.append(n)
    
        # Construct output nodes.
        for i in range(num_output_nodes):
            n = HiddenOrOutputNode(hidden_nodes)
            self._output_nodes.append(n)

        self._hidden_or_output_nodes = hidden_nodes + self._output_nodes

    def evaluate(self, vector):
        """Given an input vector, generate the corresponding output vector
        the current weights.
        """
       
        # Load the vector into the input nodes.
        for node, x in zip(self._input_nodes, vector):
            node.output = x

        # Starting with the first hidden node and moved "forward" through the
        # network, request that each node calculate its output (given its
        # current weights).
        map(HiddenOrOutputNode.calc_output, self._hidden_or_output_nodes)

        # Collect the output values of all output nodes as a vector.
        outputs = map(lambda n: n.output, self._output_nodes)

        return outputs


def error(actual_output, target_outout):
    e = 0
    for actual, target in zip(actual_output, target_output):
        e += math.pow(target - actual, 2)
    return e / 2

training_set = ([3, 7, 12], [1, 0])
nw = Network(num_input_nodes=3, num_hidden_nodes=4, num_output_nodes=2)

input_vector, target_output = training_set
actual_output = nw.evaluate(input_vector)
e = error(actual_output, target_output)
print e

