"""Convert the 'Digit Recogniser' training set CSV file provided by Kaggle
into a format compatible with the Artificial Neural Network.
"""

def fmt(ln):

    ln = ln.split(",")

    i = map(lambda x: float(x) / 255, ln[1:])
    i = map(str, i)

    o = [0] * 10
    o[int(ln[0])] = 1
    o = map(str, o)

    return ','.join(i) + ":" + ','.join(o)

with open("../train/digit.train", "w") as o:
    skipped_header = False
    for ln in open("../train/digit.train.csv"):
        if not skipped_header:
            skipped_header = True
            continue
        o.write(fmt(ln)+'\n')

