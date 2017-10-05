import sys
from math import radians,degrees
import numpy as np
import numpy.linalg as lin
from numpy.polynomial import polynomial as P

#   single dim >= 2

def isherm(a):
    m = np.asmatrix(a)
    return m.H == m

def isdiag(a):
    nonzeros = np.count_nonzero(a - np.diag(np.diagonal(a)))
    return nonzeros == 0

def istriu(a):

    n = a.shape[0]
    max_offset = n-1

    for i in np.arange(0,max_offset+1):
        a = a - np.diag(np.diagonal(a), k=i)

    return np.count_nonzero(a) == 0

def istril(a):
    n = a.shape[0]
    max_offset = n - 1

    for i in np.arange(-1,-1*max_offset-1,-1):
        a = a - np.diag(np.diagonal(a), k=i)

    return np.diag(a) == np.ones(n) and np.count_nonzero(a) == n

def isSym(a):
    h = np.transpose(a) == a
    return h

def isSkewSym(a):
    h = np.transpose(a) == -1*a
    return h

def iseye(a):
    return np.diagonal(a) == np.zeros((a.shape[0]))

def isunitary(a):
    return np.matmul(a,np.concatenate(a)) == np.eye(a.shape[0])

def isrot(*args):

    r = args[0]

    h = np.abs(lin.det(r) - 1) < np.finfo(float).eps

    if len(args) == 2:

      h = h and [args[1],args[1]] == r.shape[0:2]

    return h

def ishomog(*args):
    tr = args[0]


    h = np.abs(lin.det(tr[0:3,0:3])-1) < np.finfo(float).eps

    return h

def isbox(*args):
    a = args[0]

    dim = a.shape

    h = len(np.unique(dim)) == 1

    if len(args) == 2:

        dim_test = args[1]

        h = h and a.ndim == dim_test

    return h

#   single dim = 1

def iscvec(a):

    return a.shape[1] == 1

def isrvec(a):

    return a.shape[0] == 1

def isunit(a):
    pass

def isarray(a):
    pass

def is2d(a):
    return a.ndim == 2

def is1d(a):
    pass

def isvec(a,l=3):


    return iscvec(a) or isrvec(a)

def isSize(a,l):

    return len(a) == l

def islen():
    pass

#  single list

def isempty(a):
    return isinstance(a,list) and len(a) == 0

#  are

def areLi():
    pass

def areCoincident():
    pass

def areOrtho(*args):
    if len(args) == 2:
        v1 = args[0]
        v2 = args[1]

        h = np.dot(v1, v2) == 0

    if len(args) == 1:

        n = len(args)

        comb = list(combinations(args[0], 2))

        for i in range(len(comb) + 1):
            h = np.arange(len(comb))

            h[i] = np.dot(comb[i][0], comb[i][1]) == 0

            return np.all(h)

def areOrthoNorm(*args):
    if len(args) == 2:
        v1 = args[0]
        v2 = args[1]

        h = np.dot(v1, v2) == 0

    if len(args) == 1:

        n = len(args)

        perm = list(combinations(args[0], 2))

        for i in range(len(perm)):
    pass
