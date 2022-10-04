import numpy as np
import scipy.special
from various_curve_utils import *

# confirmation
import matplotlib.pyplot as plt


class BezierCurve(object):

    def __init__(self, control_points):
        self.control_points = control_points

    def berstein(self, n, i, t):

        return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

    def bezier(self, n, t, q):
        p = calc_bezier(n, t, q)

        p = np.zeros(2)

        for i in range(n + 1):
            p += self.berstein(n, i, t) * q[i]

        return p

    def run(self):

        bezier_curve = calc_total_bezier(self.control_points)

        return bezier_curve


if __name__ == '__main__':
    control_points = [[30, 30], [20, 40], [50, 50], [0, 55], [30, 70]]
    bc = BezierCurve(control_points)
    results = bc.run()

    results = np.array(results)

    plt.figure()
    plt.plot(results[:, 0], results[:, 1])
    plt.show()

