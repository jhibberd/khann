"""Load the Kaggle 'Digit Recogniser' dataset into a mongoDB database, in a 
format compatible with the neural network.

Source: http://www.kaggle.com/c/digit-recognizer
"""

from pymongo import MongoClient


def to_doc(ln):
    ln = ln.split(",")
    iv = map(lambda x: float(x) / 255, ln[1:])
    ov = [0.0] * 10
    ov[int(ln[0])] = 1.0
    return {
        "iv": iv,
        "ov": ov,
        }

if __name__ == "__main__":

    coll = MongoClient().khann_num.training
    skipped_header = False
    for ln in open("digit.training.csv"): 

        if not skipped_header:
            skipped_header = True
            continue

        doc = to_doc(ln)
        coll.save(doc)

