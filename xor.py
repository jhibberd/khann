"""Basic implementation of an artificial neural network in python using a
sigmoid activation function and the backpropagation learning algorithm.

The network attemps to learn the behaviour of an XOR gate.

The theory used for the implementation is from:
"Machine Learning" by Tom M. Mitchell (ISBN: 0-07-115467-1)

"""
import math
import numpy as np
import random


class ANN(object):
    """Implementation of an artificial neural network using a sigmoid 
    activation function and the backpropagation learning algorithm.
    """

    def __init__(self, topology, learning_rate=0.05):
        """Construct an ANN containing a specific number of input, hidden and
        output nodes.

        Multi-dimensional arrays are defined to hold the network outputs (o),
        weights (w) and error terms (e).

        Args:
            topology: ([int]) Decription of the topology of the network.
                
                Eg: [2, 2, 1] describes a network with 2 input nodes, 2 hidden
                nodes and 1 output node.

            learning_rate: (float) The learning rate of the backpropagation
                algorith.
        """

        self._o = map(lambda n: np.zeros(n), topology)
        self._e = map(lambda n: np.zeros(n), topology)

        self._w = []
        for l in range(1, len(topology)):
            def weights_for_node(_):
                return map(lambda _: random.random()-.5, range(topology[l-1]))
            layer = map(weights_for_node, range(topology[l]))
            self._w.append(layer)
        self._w.insert(0, None) # The input layer doesn't have weights.

        self._topology = topology
        self._learning_rate = learning_rate

    def train(self, training_set):
        """Train the network using a complete training set.

        Args:
            training_set: ([(input_vector, target_ouput)]) Training set.

        Returns:
            (float) Accumulated error for complete training set.

        """
        e = 0.0
        for input_vector, target_output in training_set:
            o = self.evaluate(input_vector)
            e += self._error(o, target_output)
            self._learn(target_output)
        return e

    def evaluate(self, xs):
        """Return the output vector for an input vector, given the current
        weights.

        First the input vector is loaded into the input nodes, then moving
        forward through the network update the output value of each node.
        Finally the output values belonging to the final layer are returned
        as the output vector.

        Args:
            xs: (list) Input vector.

        Returns:
            (list) Output vector.

        See: p. 97 (4.12)
        """
  
        self._o[0] = np.array(xs)

        for l in range(1, len(self._topology)):
            for i in range(self._topology[l]):
                o = np.dot(self._w[l][i], self._o[l-1])
                self._o[l][i] = self._sigmoid(o)

        return self._o[-1]
    
    def _learn(self, target_outputs):
        """Optionally called after 'evaluate' (once the outputs for an input
        vector have been set). The weights are adjusted to better produce
        the 'target_outputs' on classifying the same input vector again.

        Args:
            target_outputs: ([float]) Target outputs.

        See: p. 98 (T4.5)
        """

        # Calculate the error terms for the output nodes by comparing the 
        # target output with the actual output.
        for i,t in enumerate(target_outputs):
            o = self._o[-1][i]
            self._e[-1][i] = o * (1-o) * (t-o)

        # Calculate the error terms for all hidden nodes by moving backwards
        # through the network (starting with the layer just before the output
        # layer) and using the dot product of the weights from the next level
        # that connect upstream to the node and the error terms from the next 
        # level.
        for l in range(len(self._topology)-2, 0, -1):
            for i in range(self._topology[l]):
                w = map(
                    lambda j: self._w[l+1][j][i], range(self._topology[l+1]))
                e = np.dot(w, self._e[l+1])
                o = self._o[l][i]
                self._e[l][i] = o * (1-o) * e

        # Update weights by moving forward through the network (starting with
        # the second layer), calculating the weight delta for a given node by
        # multiplying the weight between the node and each upstream node by the
        # current output value from each upstream node (and the learning rate).
        for l in range(1, len(self._topology)):
            for i in range(self._topology[l]):
                for j in range(self._topology[l-1]):
                    delta = \
                        self._learning_rate * self._e[l][i] * self._o[l-1][j]
                    self._w[l][i][j] += delta

    def _sigmoid(self, x):
        """The 'squashing function' that produces an S-shaped output from the
        activation functioning allowing the network to model complexity beyond
        liner separability.

        Args:
            x: (float) Output from activation function before being 'squashed'.

        Returns:
            (float) 'Squashed' x.

        See: p. 97 (4.12)
        """
        return 1 / (1 + math.exp(-x))

    def _error(self, actual_output, target_output):
        """Return the error value between the actual and target outputs for a
        single training example.

        Args:
            actual_output: ([float]) Actual output.
            target_output: ([float]) Target output.

        Returns:
            (float) Error value.

        See: p. 97 (4.13) (modified)
        """
        e = 0
        for t,o in zip(target_output, actual_output):
            e += (t-o)**2
        return e * 0.5


if __name__ == "__main__":

    # Initialise the artificial neural network.
    training_set = [
        ([0, 0], [0]),
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([1, 1], [0]),
        ]
    ann = ANN(
        topology=[2, 2, 1],
        learning_rate=0.05)

    # Iteratively learn from the training set until an acceptable error value 
    # is obtained.
    e = float('inf')
    i = 0
    while e > .2:
        e = ann.train(training_set)
        if i % 100 == 0:
            print "Error %s" % e
        i += 1

    # Display learnt XOR classification.
    for x, _ in training_set:
        print "%s -> %s" % (x, ann.evaluate(x))

