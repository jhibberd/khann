from numpy import arange

E = 0.8 # max error
a = 0.5 # original LR
def f(e):
    return e ** (6 * (E-e))

for x in arange(.5, 0, -.05):
    print "%.2f\t%s" % (x, f(x))
