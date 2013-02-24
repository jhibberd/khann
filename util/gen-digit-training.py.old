"""Convert the 'Digit Recogniser' training set CSV file provided by Kaggle
into a format compatible with the Artificial Neural Network.
"""

# Originally 28x28
def compress(vs):

    # Linear array matric
    matrix = {}
    x = y = 0
    for v in vs:
        matrix[(x, y)] = v
        if x == 27:
            x = 0
            y += 1
        else:
            x += 1

    """
    # Render graph
    for y in range(0, 28):
        ln = []
        for x in range(0, 28):
            if matrix[(x, y)] > 0:
                ln.append("X")
            else:
                ln.append(" ")
        print ''.join(ln)
    """

    # Compress
    RATIO = 2
    matrix_c = {}
    for y in range(0, 28 / RATIO):
        for x in range(0, 28 / RATIO):
            
            vs = []
            for x2 in range(x*RATIO, (x*RATIO)+RATIO):
                for y2 in range(y*RATIO, (y*RATIO)+RATIO):
                    v = matrix.get((x2, y2))
                    if v != None:
                        vs.append(v)
            
            v = float(sum(vs)) / float(len(vs))

            matrix_c[(x, y)] = v
    
    """
    # Render compressed graph
    for y in range(0, 28 / RATIO):
        ln = []
        for x in range(0, 28 / RATIO):
            if matrix_c[(x, y)] > 0:
                ln.append("X")
            else:
                ln.append(" ")
        print ''.join(ln)

    print "Press to continue"
    raw_input()
    """

    # To linear
    ln = []
    for y in range(0, 28 / RATIO):
        for x in range(0, 28 / RATIO):
            ln.append(matrix_c[(x, y)])
    return ln

def fmt(ln):

    ln = ln.split(",")

    i = map(lambda x: float(x) / 255, ln[1:])
    #i = compress(i)
    i = map(str, i)

    o = [0] * 10
    o[int(ln[0])] = 1
    o = map(str, o)

    return ','.join(i) + ":" + ','.join(o)

with open("../data/digit.training", "w") as o:
    skipped_header = False
    for ln in open("../data/digit.training.csv"): 
        if not skipped_header:
            skipped_header = True
            continue
        o.write(fmt(ln)+'\n')

