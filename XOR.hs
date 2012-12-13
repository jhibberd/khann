import Data.Array
import Data.List
import Debug.Trace
import System.Random

type Outputs =      Array (Int, Int) Float
type ErrorTerms =   Array (Int, Int) Float
type Weights =      Array (Int, Int, Int) Float
data Network =      Network Outputs ErrorTerms Weights

--instance Show Network where
--    show (Network a b c) = "\n" ++ show a ++ "\n" ++ show b ++ "\n" ++ show c

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
rangeWeights = concat . map rangeWeights' $ range (1, t)

-- | Return range for all used weight elements in a layer.
rangeWeights' :: Int -> [(Int, Int, Int)]
rangeWeights' l = concat . map (rangeWeights'' l) $ rangeLevel l

-- | Return range for all used weight elements for a node in a layer.
rangeWeights'' :: Int -> Int -> [(Int, Int, Int)]
rangeWeights'' l i = [(l, i, j) | j <- rangeLevel (l-1)]

-- | Given the topology return the indices available at a level.
rangeLevel :: Int -> [Int]
rangeLevel l = range (0, (topology !! l)-1)

t = (length topology) -1
m = (maximum topology) -1
boundsOutputs =     ((0, 0),    (t, m))
boundsErrorTerms =  ((0, 0),    (t, m))
boundsWeights =     ((0, 0, 0), (t, m, m))

topology = [2, 2, 1]
trainingSet = [
              ([0, 0], [0]),
              ([1, 0], [1]),
              ([0, 1], [1]),
              ([1, 1], [0])
              ]

-- TODO At some point move from Float to Double.
-- | The Sigmoid function.
sigmoid :: Float -> Float
sigmoid x = 1.0 / (1 + exp (-x))

-- | Dot product of two vectors.
dot :: Num a => [a] -> [a] -> a 
dot xs ys = sum (zipWith (*) xs ys)

setOutputs :: [Float] -- Input vector
           -> Network
           -> Network
setOutputs xs (Network os es ws) = Network os' es ws
    where os' =             foldl' layer input (range (1, t))
          input =           os // [((0, i), x) | (i, x) <- zip [0..] xs]
          layer os l =      os // [((l, i), calc os l i) | i <- rangeLevel l] 
          calc os l i =     sigmoid $ dot (weights l i) (upstream os l)
          weights l i =     [ws!(l, i, j) | j <- [0..m]]
          upstream os l =   [os!(l-1, j)  | j <- [0..m]]

setErrorTerms :: [Float] -- Target output
              -> Network 
              -> Network
setErrorTerms ts n@(Network os es ws) = Network os es' ws
    where es' = foldl' f output (reverse $ range (1, (t-1))) 
          f es l = es // [((l, i), g es l i) | i <- range (0, (topology !! l)-1)]
          g es l i = let e = dot (weights (l+1) i) (errorTerms es (l+1))
                         o = os!(l, i)
                     in o * (1-o) * e
          weights l i =     [ws!(l, j, i) | j <- range (0, (topology !! l)-1)]
          errorTerms es l = [es!(l, j) | j <- range (0, (topology !! l)-1)]
          output = setOutputErrorTerms ts n

setOutputErrorTerms :: [Float] -- Target output
                    -> Network
                    -> ErrorTerms
setOutputErrorTerms ts (Network os es ws) = es'
    where es' = es // [((t, i), calc (os!(t, i)) t') | (i, t') <- zip [0..] ts] 
          calc o t = o * (1-o) * (t-o)

learningRate :: Float -- Error
             -> Float
learningRate e = 0.5 --e / (2 ** e) -- TODO Need to be more quadratic

-- | Adjust the network weights according to a learning rate.
setWeights :: Float -> Network -> Network
setWeights lr (Network os es ws) = Network os es ws'
    where ws' = ws // [(i, f i) | i <- rangeWeights]
          f (l, i, j) = ws!(l, i, j) + (lr * es!(l, i) * os!(l-1, j))

errorVal :: [Float] -- Actual output
         -> [Float] -- Target output
         -> Float -- Error
errorVal os ts = (sum $ zipWith (\t o -> (t-o)**2) ts os) * 0.5

output :: Network -> [Float]
output (Network os _ _) = [os!(t, i) | i <- range (0, (topology !! t)-1)]

train :: Network -> (Network, Float) -- Includes error
train n = foldl' f (n, 0) trainingSet
    where f (n, e) (x, t) = let n' = setErrorTerms t $ setOutputs x n
                                o = output n'
                                e' = errorVal o t
                                n'' = setWeights (learningRate e') n'
                            in (n'', e+e')

main = do
    n <- initNetwork
    n' <- iteration n
    test n'
    return ()

iteration :: Network -> IO Network
iteration n = do
    let (n'@(Network _ _ c'), e) = train n
    putStrLn (show e ++ "\t" ++ show (learningRate e)) 
    if e < 0.15
           then return n'
           else iteration n'

test :: Network -> IO ()
test n = do
    let output = map f trainingSet
    mapM_ putStrLn output
    where f (x, t) = let o = output $ setOutputs x n
                     in (show x ++ "=" ++ show t ++ " => " ++ show o)

