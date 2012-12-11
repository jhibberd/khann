import Data.Array
import Data.List
import Debug.Trace
import System.Random

type Outputs = Array (Int, Int) Float
type ErrorTerms = Array (Int, Int) Float
type Weights = Array (Int, Int, Int) Float
data Network = Network Outputs ErrorTerms Weights

instance Show Network where
    show (Network a b c) = "\n" ++ show a ++ "\n" ++ show b ++ "\n" ++ show c

initNetwork :: IO Network
initNetwork = do
    w' <- w
    return (Network os es (ws' w'))
    where w = sequence $ map (\_ -> randomRIO (-0.5, 0.5)) (range boundsWeights)
          os = emptyArray2
          es = emptyArray2
          ws' ws = array boundsWeights $ zip (range boundsWeights) ws

emptyArray2 = array boundsOutputs [(i, 0) | i <- range boundsOutputs]
emptyArray3 = array boundsWeights [(i, 0) | i <- range boundsWeights]

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
          layer os l =      os // [((l, i), calc os l i) | 
                                   i <- range (0, (topology !! l)-1)] 
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
          weights l i =     [ws!(l, i, j) | j <- [0..m]]
          errorTerms es l = [es!(l, j) | j <- [0..m]]
          output = setOutputErrorTerms ts n

setOutputErrorTerms :: [Float] -- Target output
                    -> Network
                    -> ErrorTerms
setOutputErrorTerms ts (Network os es ws) = es'
    where es' = es // [((t, i), calc (os!(t, i)) t') | (i, t') <- zip [0..] ts] 
          calc o t = o * (1-o) * (t-o)

learningRate = 0.05

setWeights :: Network -> Network
setWeights (Network os es ws) = Network os es ws'
    where ws' = ws // [((l, i, j), f l i j) | l <- range (1, t), i <- range (0, m), j <- range (0, m)]
          f l i j = let delta = learningRate * es!(l, i) * os!(l-1, j)
                    in ws!(l, i, j) + delta


errorVal :: [Float] -- Actual output
         -> [Float] -- Target output
         -> Float -- Error
errorVal os ts = (sum $ zipWith (\t o -> (t-o)**2) ts os) * 0.5

-- TODO Not generic
output :: Network -> [Float]
output (Network os _ _) = [os!(t, 0)]

train :: Network -> (Network, Float) -- Includes error
train n = foldl' f (n, 0) trainingSet
    where f (n, e) (x, t) = let n' = setWeights . setErrorTerms t $ setOutputs x n
                                o = output n'
                                e' = errorVal o t
                            in (n', e+e')

main = do
    n <- initNetwork
    putStrLn (show n)
    n' <- iteration n
    test n'
    return ()

iteration :: Network -> IO Network
iteration n = do
    let (n'@(Network _ _ c'), e) = train n
    --putStrLn (show n')
    putStrLn (show e)
    if e < 0.3
           then return n'
           else iteration n'

test :: Network -> IO ()
test n = do
    let output = map f trainingSet
    mapM_ putStrLn output
    where f (x, t) = let o = output $ setOutputs x n
                     in (show x ++ "=" ++ show t ++ " => " ++ show o)

