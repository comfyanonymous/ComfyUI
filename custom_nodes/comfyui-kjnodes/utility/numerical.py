from functools import reduce
from itertools import cycle
from math import factorial

import numpy as np
import scipy.sparse as sp


def difference(derivative, accuracy=1):
    # Central differences implemented based on the article here:
    # http://web.media.mit.edu/~crtaylor/calculator.html
    derivative += 1
    radius = accuracy + derivative // 2 - 1
    points = range(-radius, radius + 1)
    coefficients = np.linalg.inv(np.vander(points))
    return coefficients[-derivative] * factorial(derivative - 1), points


def operator(shape, *differences):
    # Credit to Philip Zucker for figuring out
    # that kronsum's argument order is reversed.
    # Without that bit of wisdom I'd have lost it.
    differences = zip(shape, cycle(differences))
    factors = (sp.diags(*diff, shape=(dim,) * 2) for dim, diff in differences)
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)