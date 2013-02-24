"""Script to generate training set for all possible additions of two binary
numbers (0-31) and their binary solution.
"""

from pymongo import MongoClient
from itertools import product


def gen():
    x = range(0, 31)
    return product(reversed(x), x)

def solve(pairs):
    for a, b in pairs:
        yield a, b, a+b 

def to_bin(solutions):

    def bin_array(x, w):
        x = bin(x)[2:]
        x = list(x)
        x = (['0'] * (w-len(x))) + x
        x = map(float, x)
        return x

    for a, b, c in solutions:
        yield bin_array(a, 5) + bin_array(b, 5), bin_array(c, 6)

def to_doc(xs):
    res = []
    for x in xs:
        iv, ov = x
        res.append({
            "iv":   iv,
            "ov":   ov,
            })
    return res

if __name__ == "__main__":

    coll = MongoClient().khann_binadd.training
    xs = gen()
    xs = solve(xs)
    xs = to_bin(xs) 
    xs = to_doc(xs)
    for x in xs:
        coll.save(x)    

