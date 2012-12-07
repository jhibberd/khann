import Debug.Trace
import System.Random

-- | Config --------------------------------------------------------------------

trainingSet :: [([Float], [Float])]
trainingSet = [
              ([0, 0], [0]),
              ([1, 0], [1]),
              ([0, 1], [1]),
              ([1, 1], [0])
              ]

topology = [2, 2, 1]

learningRate = 0.5

-- Init ANN --------------------------------------------------------------------

type Outputs = [[Float]]
type ErrorTerms = [[Float]]
type Weights = [[[Float]]]
data Network = Network Outputs ErrorTerms Weights

instance Show Network where
    show (Network o e w) = "Outputs:\t" ++ show o ++ "\n" ++
                           "Error terms:\t" ++ show e ++ "\n" ++
                           "Weights:\t" ++ show w

initOutputs :: Outputs
initOutputs = map (`replicate` 0) topology

-- TODO Input later has no error terms, should be []
initErrorTerms :: ErrorTerms
initErrorTerms = map (`replicate` 0) topology

initWeights :: IO Weights
initWeights = sequence $ ( return [[]] : map layer [1..length topology -1] )
    where layer l = sequence $ [weights l | _ <- [1..topology !! l]]
          weights l = sequence $ [w | _ <- [1..topology !! (l-1)]]
          w = randomRIO (-0.5, 0.5)

initNetwork :: IO Network
initNetwork = do
    w <- initWeights
    let o = initOutputs
        e = initErrorTerms
    return (Network o e w)

-- Learning --------------------------------------------------------------------

-- TODO At some point move from Float to Double.
-- | The Sigmoid function.
sigmoid :: Float -> Float
sigmoid x = 1.0 / (1 + exp (-x))

-- | Dot product of two vectors.
dot :: Num a => [a] -> [a] -> a 
dot xs ys = sum (zipWith (*) xs ys)

-- TODO Reduce variable names but provide detailed description to compensate.
evaluate :: Network -> [Float] -> Network
evaluate (Network os es ws) xs = Network os'' es ws
    where os'' = foldl f [xs] [1..length topology -1]
          f os' l = os' ++ [[output os' l i | i <- [0..(topology !! l)-1]]]
          output os' l i' = sigmoid $ dot nodeWeights upstreamOutputs
              where nodeWeights = (ws !! l) !! i'
                    upstreamOutputs = os' !! (l-1)


errorTerms :: Network 
           -> [Float] -- Target outputs
           -> Network
errorTerms n@(Network os es ws) ts = Network os ([[]] ++ otherEs) ws
    where otherEs = foldl f [(outputET n ts)] [start, (start-1)..1]
              where f es' l = [[g es' l i | i <- [0..(topology !! l)-1]]] ++ es'
                    start = length topology-2
                    ws' l i  = [ws !!!! (l+1, j, i) | j <- [0..(topology !! (l+1))-1]]
                    g es' l i = let e = dot (ws' l i) (head es')
                                    o = os !!! (l, i)
                                in o * (1-o) * e

-- | Return error terms for the output later of a network.
outputET :: Network 
         -> [Float] -- Target outputs
         -> [Float]
outputET (Network os es ws) ts = map e (enum ts)
    where e (i, t) = let o = (last os) !! i
                     in o * (1-o) * (t-o)

--hiddenET :: Network ->

-- | Augment a list with the index position of each element.
enum :: [a] -> [(Int, a)]
enum = zip [0..]

(!!!) :: [[a]] -> (Int, Int) -> a
xs !!! (i1, i2) = (xs !! i1) !! i2

(!!!!) :: [[[a]]] -> (Int, Int, Int) -> a
xs !!!! (i1, i2, i3) = ((xs !! i1) !! i2) !! i3

{-

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

-}
-- TODO Implement "learn" as two functions: one to update error terms; another
-- to update the weights.


main = do
    net <- initNetwork
    print net
    let net1 = evaluate net [1, 1]
    print net1
    let net2 = errorTerms net1 [0]
    print net2
    return ()

