import numpy as np
import sys
sys.path.append("..")
import nqpSover

A = np.matrix('5,-2,-1;-2,4,3;-1,5,3')
b = np.array([2,-35,-47])
c = np.array([0,0,0])[np.newaxis]


res = nqpSover.solve(A,b)
