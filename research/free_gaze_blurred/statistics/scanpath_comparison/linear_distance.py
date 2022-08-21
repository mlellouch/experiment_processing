
from scipy.spatial.distance import cdist

"""
Linear distance
Given two scanpaths f and g with n1 and n2 fixations,
compute the distance d1i between the ith fixation fi and its nearest neighbor fixation in g
and the distance d2j between the jth fixation gj with its nearest neighbor fixation in f.
Then the similarity S of the scanpaths f and g is defined as
D = the sum of d1i (i = 1,...,n1) + sum of d2j (j = 1,...,n2) / max(n1, n2)
"""


# calculates linear distance measure, giving the choice to normalize the result
def calc_linear_distance(fixations1, fixations2, normalize=True):
    distances = cdist(fixations1, fixations2)
    d = distances.min(axis=0).sum() + distances.min(axis=1).sum()
    if normalize:
        d /= max(len(fixations1), len(fixations2))
    return d