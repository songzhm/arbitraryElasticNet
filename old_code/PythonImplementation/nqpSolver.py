import numpy as np
from math import sqrt
import copy

class nqpSover(object):

    def solve(self,Amat,bvec,err_tol = 1e-8):
        A_plus = copy.deepcopy(Amat)
        A_plus[A_plus<0]=0

        A_minus = copy.deepcopy(Amat)
        A_minus[A_minus>0]=0
        A_minus = abs(A_minus)

        v = np.array([1.0 for x in range(len(bvec))])

        updateFactor = np.array([0.0 for x in range(len(bvec))])

        while(((updateFactor*v-v)**2).sum()>err_tol):
            dFa = np.array(A_plus.dot(v))[0]
            dFb = copy.deepcopy(bvec)
            dFc = np.array(A_minus.dot(v))[0]
            for i in range(len(bvec)):
                updateFactor[i] = \
                    float((-dFb[i]+sqrt(dFb[i]**2+4*dFa[i]*dFc[i]))/(2*dFa[i]))

            if np.count_nonzero(~np.isnan(updateFactor)) ==len(bvec):
                v = updateFactor * v
            else:
                break
        return v

A = np.mat('5,-2,-1;-2,4,3;-1,5,3')
b = np.array([2,-35,-47])

s = nqpSover()
res = s.solve(A,b)

print(res)
