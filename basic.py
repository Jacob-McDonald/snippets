from numpy import array,ones,kron,ndarray
from scipy.linalg import norm

def unit(v):
    n = norm(v, 'fro')
    u = v / n
    return u

def numcols(a):
    return a.shape[1]

def numrows(a):
    return a.shape[0]

def skew(v):
    if len(v) == 3:
        s = array([0,-v[0],v[1],v[2],0,-v[0],-v[1],v[0],0]).reshape((3,3))

    elif len(v) == 1:
        s = array([0,-v,v,0]).reshape((2,2))

    return s

def repmat(a,*args):

    b = args[0]

    if len(args) == 2:

        if isinstance(b, int):

            rmat = kron(ones((b, b)), a)

        if isinstance(b, (list,ndarray)):

            rmat = kron(ones(b), a)

    return rmat

def angdiff(*args):

    th1 = args[0]
    th2 = args[1]

    if len(args) == 1:

        if th1.len == 2:
            d = th1[0] - th1[1]
        else:
            d=th1

    if len(args) == 2:
        if len(th1) > 1 and len(th2) > 1:
            assert len(th1) == len(th2),'vectors must be same shape'

        d = th1 - th2

    d = np.mod(d + np.pi, 2*np.pi)-np.pi

    return d

def proj(a, dir):
    proj = np.dot(a, dir) * dir / np.square(np.linalg.norm(dir))

    return proj

def cycle(n, index):
    array = np.arange(1, n + 1)

    index =  np.mod((index -1),n)

    return array[index]

def points2plane(v, form='general', cat = False):
    n = len(v)

    system_a = np.vstack(v)

    new_column = np.array([-1 * np.ones(n)]).T

    system_ac = np.concatenate((system_a, new_column), axis=1)

    [p, l, u] = la.lu(system_ac)

    a = u[-1, -2]

    b = u[-1, -1]

    c = a / -b

    newRow = np.zeros(n + 1).reshape((1, n + 1))

    newRow[0, -1] = 1

    system_ac2 = np.concatenate((system_ac, newRow))

    system_b = np.zeros(n + 1)

    system_b[-1] = c

    sol = np.linalg.solve(system_ac2, system_b)

    if cat == False:

        coeff = sol[:-1]

        c = sol[-1]

        return [coeff, c]

    elif cat == True:

        if form == 'standard':
            sol[-1] = -1 * sol[-1]

        return sol