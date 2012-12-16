import Data.Array
import Data.List
import Debug.Trace
import System.Random

type InputVector =          [Float]
type TargetOutputVector =   [Float]
type ActualOutputVector =   [Float]
type Outputs =              Array (Int, Int) Float
type ErrorTerms =           Array (Int, Int) Float
type Weights =              Array (Int, Int, Int) Float
data Network =              Network Outputs ErrorTerms Weights

-- Config ----------------------------------------------------------------------

topology = [2, 2, 1]
trainingSet = [
              ([0, 0], [0]),
              ([1, 0], [1]),
              ([0, 1], [1]),
              ([1, 1], [0])
              ]
threshold = 0.1

-- Intialise Network -----------------------------------------------------------

initNetwork :: IO Network
initNetwork = do
    rs' <- rs
    return (Network os es (ws rs'))
    where rs = sequence $ map (\_ -> randomRIO (-0.5, 0.5)) rangeWeights
          os = listArray boundsOutputs [0,0..] 
          es = listArray boundsErrorTerms [0,0..]
          ws rs = listArray boundsWeights [0,0..] // zip rangeWeights rs

-- | Return complete range for all used weight elements.
rangeWeights :: [(Int, Int, Int)]
rangeWeights = concat . map rangeWeights' $ range (1, lastLayer)

-- | Return range for all used weight elements in a layer.
rangeWeights' :: Int -> [(Int, Int, Int)]
rangeWeights' l = concat . map (rangeWeights'' l) $ rangeLevel l

-- | Return range for all used weight elements for a node in a layer.
rangeWeights'' :: Int -> Int -> [(Int, Int, Int)]
rangeWeights'' l i = [(l, i, j) | j <- rangeLevel (l-1)]

-- | Given the topology return the indices available at a level.
rangeLevel :: Int -> [Int]
rangeLevel l = range (0, (topology !! l)-1)

lastLayer =         (length topology) -1
maxLayerWidth =     (maximum topology) -1
boundsOutputs =     ((0, 0),    (lastLayer, maxLayerWidth))
boundsErrorTerms =  ((0, 0),    (lastLayer, maxLayerWidth))
boundsWeights =     ((0, 0, 0), (lastLayer, maxLayerWidth, maxLayerWidth))

-- Network Views ---------------------------------------------------------------

-- | Return the weights on level 'l' for node 'i'.
weightsAt :: Array (Int, Int, Int) Float -> Int -> Int -> [Float]
weightsAt ws l i = map (ws!) (rangeWeights'' l i)

-- Return the outputs at layer 'l'.
outputsAt :: Outputs -> Int -> [Float]
outputsAt os l = [os!(l, i) | i <- rangeLevel l]

-- Return the outputs at layer 'l' given a network.
outputsAt' :: Int -> Network -> [Float]
outputsAt' l (Network os _ _) = outputsAt os l

-- Return the range of output indices at layer 'l'.
rngOutputs :: Int -> [(Int, Int)]
rngOutputs l = [(l, i) | i <- rangeLevel l]

-- Return the range of error term indices at layer 'l'.
rngErrorTerms :: Int -> [(Int, Int)]
rngErrorTerms l = [(l, i) | i <- rangeLevel l]

-- Learning --------------------------------------------------------------------

-- | Given an input vector, move forward through the network setting the output
-- value for each node.
setOutputs :: InputVector -> Network -> Network
setOutputs xs (Network os es ws) = Network os' es ws
    where input =       os // zip (rngOutputs 0) xs
          os' =         foldl' layer input [1..lastLayer]
          layer os l =  os // [((l, i), f os l i) | i <- rangeLevel l] 
          f os l i =    sigmoid $ dot (weightsAt ws l i) (outputsAt os (l-1))

-- | Moving backwards through the network set the error term for each node
-- based on its contribution towards the difference between the actual output
-- and target output vectors.
setErrorTerms :: TargetOutputVector -> Network -> Network
setErrorTerms ts n@(Network os es ws) = Network os es' ws
    where es' = foldl' f output [lastLayer-1,lastLayer-2..1] 
          f es l = es // [((l, i), g es l i) | i <- rangeLevel l]
          g es l i = let e = dot (weights (l+1) i) (errorTerms es (l+1))
                         o = os!(l, i)
                     in o * (1-o) * e
          weights l i =     [ws!(l, j, i) | j <- rangeLevel l]
          errorTerms es l = [es!(l, j) | j <- rangeLevel l]
          output = setOutputErrorTerms ts n

-- | Set the error terms for the output layer.
setOutputErrorTerms :: TargetOutputVector -> Network -> ErrorTerms
setOutputErrorTerms ts (Network os es _) = es'
    where es' = es // zip (rngErrorTerms lastLayer) 
                      (map f (zip (outputsAt os lastLayer) ts)) 
          f (o, t) = o * (1-o) * (t-o)

-- | Adjust the network weights according to a learning rate.
setWeights :: Float -> Network -> Network
setWeights lr (Network os es ws) = Network os es ws'
    where ws' = ws // [(i, f i) | i <- rangeWeights]
          f (l, i, j) = ws!(l, i, j) + (lr * es!(l, i) * os!(l-1, j))

-- | Return the learning rate as a function of the error.
learningRate :: Float -> Float
learningRate e = if e > 0.2 then 0.5 else e ** (7 * (0.65-e))

-- | Given a single pair of actual and target output vectors return the 
-- associated error value.
errorVal :: [Float] -> [Float] -> Float -- Error
errorVal os ts = (sum $ zipWith (\t o -> (t-o)**2) ts os) * 0.5

-- | Train the network on each member of the training set, then return the
-- final network and its associated error.
train' :: Network -> (Network, Float)
train' n = foldl' f (n, 0) trainingSet
    where f (n, e) (x, t) = let n' = setErrorTerms t $ setOutputs x n
                                o = outputsAt' lastLayer n'
                                e' = errorVal o t
                                n'' = setWeights (learningRate e') n'
                            in (n'', e+e')

-- | Repeatedly train the network until its error, after having processed all
-- members of the training set, is below the threshold.
train :: Network -> Int -> Network
train n i =
    let (n', e) = train' n
        e' = trace i e
    in if e' < threshold then n' else train n' (i+1)
    where trace i e 
              | rem i 5000 == 0 = traceShow e e
              | otherwise =      e

-- Helpers ---------------------------------------------------------------------

-- | The Sigmoid function.
sigmoid :: Float -> Float
sigmoid x = 1.0 / (1 + exp (-x))

-- | Dot product of two vectors.
dot :: Num a => [a] -> [a] -> a 
dot xs ys = sum (zipWith (*) xs ys)

-- Main ------------------------------------------------------------------------

test :: Network -> IO ()
test n = do
    let output = map f trainingSet
    mapM_ putStrLn output
    where f (x, t) = let o = (outputsAt' lastLayer) $ setOutputs x n
                     in (show x ++ "=" ++ show t ++ " => " ++ show o)

main = do
    n <- initNetwork
    let n' = train n 0
    test n'

