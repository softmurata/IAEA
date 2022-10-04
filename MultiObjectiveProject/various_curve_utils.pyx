import numpy as np
cimport numpy as np
import cython
import scipy

cdef berstein(int n, int i, double t):

    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

cdef calc_bezier(int n, double t, np.ndarray q):
    cdef np.ndarray p
    cdef int i
    p = np.zeros(2)

    for i in range(n + 1):
        p += berstein(n, i, t) * q[i]

    return p

cpdef calc_total_bezier(list control_points):
    cdef np.ndarray q
    cdef int n
    cdef list bezier_curve
    cdef double t
    q = np.array(control_points)
    n = q.shape[0] - 1

    bezier_curve = []
    for t in np.linspace(0, 1, 500):
        bezier_curve.append(calc_bezier(n, t, q))

    return bezier_curve


