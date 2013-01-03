import Data.Array
import Data.List
import Debug.Trace
import System.Random
import System.Environment
import Data.Maybe

type InputVector =          [Float]
type TargetOutputVector =   [Float]
type ActualOutputVector =   [Float]
type Outputs =              Array (Int, Int) Float
type ErrorTerms =           Array (Int, Int) Float
type Weights =              Array (Int, Int, Int) Float
type TrainingSet =          [([Float], [Float])]
data Network =              Network Outputs ErrorTerms Weights

-- Config ----------------------------------------------------------------------

-- topology = [10, 15, 10, 6] -- binadd
topology = [784, 1176, 392, 10] -- digit
threshold = 0.1
learningRate = 0.5

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

-- | Given a single pair of actual and target output vectors return the 
-- associated error value.
errorVal :: [Float] -> [Float] -> Float -- Error
errorVal os ts = (sum $ zipWith (\t o -> (t-o)**2) ts os) * 0.5

correctClassify :: [Float] -> [Float] -> Int
correctClassify os ts = let os' = map (fromIntegral . round) os
                        in if os' == ts then 1 else 0

-- | Train the network on each member of the training set, then return the
-- final network and its associated error.
train' :: TrainingSet -> Network -> (Network, Float, Int)
train' tset n = foldl' f (n, 0, 0) tset
    where f (n, e, c) (x, t) = let n' = setErrorTerms t $ setOutputs x n
                                   o = outputsAt' lastLayer n'
                                   e' = errorVal o t
                                   c' = correctClassify o t
                                   n'' = setWeights learningRate n'
                               in (n'', e+e', c+c')

-- | Repeatedly train the network until its error, after having processed all
-- members of the training set, is below the threshold.
train :: TrainingSet -> Network -> Int -> IO Network
train tset n i = do
    let (n'@(Network _ _ ws), e, c) = train' tset n
        e' = trace i e ws c
    if e' < threshold || areWeightsSame n n' then return n' else train tset n' (i+1)
    where trace i e ws c
              | rem i 10 == 0 = traceShow (show e ++ "-" ++ show c ++ "/" ++ show (length tset)) e
              | otherwise =     e

areWeightsSame :: Network -> Network -> Bool
areWeightsSame (Network _ _ ws1) (Network _ _ ws2) = ws1 == ws2

-- Helpers ---------------------------------------------------------------------

-- | The Sigmoid function.
sigmoid :: Float -> Float
sigmoid x = 1.0 / (1 + exp (-x))

-- | Dot product of two vectors.
dot :: Num a => [a] -> [a] -> a 
dot xs ys = sum (zipWith (*) xs ys)

-- Main ------------------------------------------------------------------------

test :: TrainingSet -> Network -> IO ()
test tset n = do
    let output = map f tset
    mapM_ putStrLn output
    where f (x, t) = let o = (outputsAt' lastLayer) $ setOutputs x n
                     in (show x ++ "=" ++ show t ++ " => " ++ show o)

trainingSet :: String -> IO TrainingSet
trainingSet fn = do
    xs <- readFile fn
    return . map fmt $ lines xs
    where fmt x = let (a, b) = splitAt (fromJust $ elemIndex ':' x) x
                      a' = map read $ splitOn ',' a
                      b' = map read . splitOn ',' $ tail b
                  in (a', b')

splitOn :: Eq a => a -> [a] -> [[a]]
splitOn d xs = case elemIndex d xs of
                   Nothing ->  [xs]
                   (Just i) -> let (x', xs') = splitAt i xs
                               in x' : splitOn d (tail xs')

main = do
    fn <- fmap head getArgs
    tset <- trainingSet fn
    n <- initNetwork
    n' <- train tset n 0
    test tset n'

