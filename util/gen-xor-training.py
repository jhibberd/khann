"""Script to generate training set for all possible inputs and outputs of an
Exclusive Or gate.
"""

from pymongo import MongoClient
from itertools import product


if __name__ == "__main__":
    coll = MongoClient().khann_xor.training
    coll.save({"iv": [1.0, 1.0], "ov": [0.0]})
    coll.save({"iv": [1.0, 0.0], "ov": [1.0]})
    coll.save({"iv": [0.0, 1.0], "ov": [1.0]})
    coll.save({"iv": [0.0, 0.0], "ov": [0.0]})

