"""Script to generate training set for all possible additions of two binary
numbers (0-31) and their binary solution.
"""

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
        return x

    for a, b, c in solutions:
        yield bin_array(a, 5) + bin_array(b, 5), bin_array(c, 6)

def to_str(xs):
    f = lambda y: ','.join(y)
    for a, b in xs:
        yield f(a) + ':' + f(b)

def output(xs):
    print '\n'.join(xs)

x = gen()
x = solve(x)
x = to_bin(x)
x = to_str(x)
output(x)

