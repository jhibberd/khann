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
          f os' l = os' ++ [[output os' l i | i <- range (topology !! l)]]
          output os' l i' = sigmoid $ dot nodeWeights upstreamOutputs
              where nodeWeights = (ws !! l) !! i'
                    upstreamOutputs = os' !! (l-1)

errorTerms :: Network 
           -> [Float] -- Target outputs
           -> Network
errorTerms n@(Network os es ws) ts = Network os ([[]] ++ otherEs) ws
    where otherEs = foldl f [(outputET n ts)] [start, (start-1)..1]
              where f es' l = [[g es' l i | i <- range (topology !! l)]] ++ es'
                    start = length topology-2
                    ws' l i  = [ws !!!! (l+1, j, i) | j <- range (topology !! (l+1))]
                    g es' l i = let e = dot (ws' l i) (head es')
                                    o = os !!! (l, i)
                                in o * (1-o) * e

-- | Return error terms for the output layer of a network.
outputET :: Network 
         -> [Float] -- Target outputs
         -> [Float]
outputET (Network os es ws) ts = map e (enum ts)
    where e (i, t) = let o = (last os) !! i
                     in o * (1-o) * (t-o)

-- | Augment a list with the index position of each element.
enum :: [a] -> [(Int, a)]
enum = zip [0..]

(!!!) :: [[a]] -> (Int, Int) -> a
xs !!! (i1, i2) = (xs !! i1) !! i2

(!!!!) :: [[[a]]] -> (Int, Int, Int) -> a
xs !!!! (i1, i2, i3) = ((xs !! i1) !! i2) !! i3

-- | Update the weights in a network given the error terms and outputs.
updateWeights :: Network -> Network
updateWeights (Network os es ws) = Network os es ws'
    where ws' = [] : [f1 l | l <- [1..length topology -1]]
          f1 l = [f2 l i | i <- range (topology !! l)]
          f2 l i = [f3 l i j | j <- range (topology !! (l-1))]
          f3 l i j = let delta = learningRate * es !!! (l, i) * os !!! (l-1, j)
                     in ws !!!! (l, i, j) + delta

-- | List of index positions up to to (but excluding) n.
range :: Int -> [Int]
range n = [0..(n-1)]

-- | Other ---------------------------------------------------------------------

-- | Return the error value between the actual and target output for a single
--   training example.
errorVal :: [Float]    -- Actual output
         -> [Float]    -- Target outout
         -> Float      -- Error
errorVal os ts = (sum $ zipWith (\t o -> (t-o)**2) ts os) * 0.5

-- | Train the network using a complete training set
train :: Network                -- Initial network
      -> [([Float], [Float])]   -- Training set
      -> (Network, Float)       -- trained network and error
train net tset = foldl f (net, 0) tset
    where f (net', e) (x, t) = let net1 = evaluate net' x
                                   net2 = errorTerms net1 t
                                   net3 = updateWeights net2
                                   o = output net3
                                   e' = errorVal o t
                               in (net3, e+e')

-- | The output layer of a network.
output :: Network -> [Float]
output (Network os _ _) = last os

{-
main = do
    net <- initNetwork
    print net
    let net1 = evaluate net [1, 1]
    print net1
    let net2 = errorTerms net1 [0]
    print net2
    let net3 = updateWeights net2
    print net3
    return ()
-}

main = do
    net <- initNetwork
    net' <- iteration net 100000
    test net'
    return ()

threshold = 0.13

iteration :: Network -> Float -> IO Network
iteration net e = do
    let (net', e') = train net trainingSet
    print e'
    if e' < threshold 
        then return net' 
        else iteration net' e'

test :: Network -> IO ()
test net = do
    let output = map f trainingSet
    mapM_ putStrLn output
    where f (x, t) = let o = output $ evaluate net x
                     in (show x ++ "=" ++ show t ++ " => " ++ show o)
