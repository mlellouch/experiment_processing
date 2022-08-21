import numpy as np
from scipy.spatial.distance import cdist


"""
Cross-recurrence analysis
Consider two fixation sequences: fi, i = 1,...,N, with fi = <xi,yi>
and gi, i = 1,...,N, with gi = <xi,yi>.
For fixation sequences of unequal length, the long sequence is truncated.
Two fixations fi and gj are cross-recurrent if they are close together, i.e., 
we define the cross-recurrence of two fixations cij as
cij = 1 if d(fi, gj) ≤ radius else 0
where d is the Euclidean distance, and radius is a given threshold.
"""


# main function that calculates crm - cross recurrence matrix
def cross_recurrence_analysis(fixations1, fixations2, radius):

    fixations1_copy = fixations1.copy()
    fixations2_copy = fixations2.copy()

    # truncate the longer list
    del fixations1_copy[len(fixations2_copy):]
    del fixations2_copy[len(fixations1_copy):]

    # calculates the distances between each two fixations fi and gj
    distances = cdist(fixations1_copy, fixations2_copy)

    # calculates crm
    crm = np.zeros((len(fixations1_copy), len(fixations2_copy)))
    for i in range(len(fixations1_copy)):
        for j in range(len(fixations2_copy)):
            crm[i][j] = 1 if distances[i][j] <= radius else 0
    return crm


"""
Cross-recurrence
Let C be the sum of recurrences, i.e., C = crm.sum() and N the number of fixation in each sequence
The cross-recurrence measure of two fixation sequences is defined as
REC = 100 * ( C / N**2)
It represents the percentage of cross-recurrent fixations,
i.e., the percentage of fixations that match (are close) between
the two fixation sequences.
"""


# calculates cross-recurrence measure
def calc_cross_recurrence(crm):
    n = len(crm)
    c = crm.sum()
    return 100 * (c / (n ** 2))


"""
Determinism
Let DL be the set of diagonal lines in the cross-recurrence matrix, 
all with a length of at least L, and let |·| denote cardinality.
The determinism measure is defined as
DET = 100 * ( |DL| / C )
It measures the percentage of cross-recurrent points that form diagonal lines 
and represents the percentage of fixation trajectories common to both fixation sequences.
That is, it quantifies the overlap of a specific sequence of fixations,
preserving the sequential information.
"""


# calculates determinism measure, giving the choice to take into account also the length of each line
# and the secondary diagonals. Here, DL = count, L = length
def calc_determinism(crm, length, by_length=True, sec_diagonal=True):

    # calculate the number (count) and sum of lengths (count_length) of diagonal lines
    count, count_length = calc_diagonal_lines(crm, length)

    # same for secondary diagonals
    if sec_diagonal:
        flipped_crm = np.fliplr(crm)
        count2, count_length2 = calc_diagonal_lines(flipped_crm, length)
        count += count2
        count_length += count_length2

    c = crm.sum()
    c = c if c != 0 else 1
    return 100 * (count_length / c) if by_length else 100 * (count / c)


"""
Laminarity
Let HL be the set of horizontal, and VL the set of vertical lines in the cross-recurrence matrix,
all with a length of at least L, and let |·| denote cardinality.
The laminarity measure is defined as
LAM = 100 * (|HL| + |VL|) / 2C`
Laminarity represents locations that were fixated in detail
in one of the fixation sequences, but only fixated briefly in the
other fixation sequence.
"""


# calculates laminarity measure, giving the choice to take into account also the length of each line
# Here, VL = count, HL = count2, L = length
def calc_laminarity(crm, length, by_length=True):

    # calculate the number (count) and sum of lengths (count_length) of vertical lines
    count, count_length = calc_vertical_lines(crm, length)

    # calculate the number (count2) and sum of lengths (count_length2) of horizontal lines
    count2, count_length2 = calc_horizontal_lines(crm, length)

    count += count2
    count_length += count_length2

    c = crm.sum()
    c = c if c != 0 else 1
    return 100 * (count_length / c) if by_length else 100 * (count / c)


"""
Center of recurrence mass
Finally, the center of recurrence mass (CORM) is defined as the distance 
of the center of gravity from the main diagonal, normalized such that the
maximum possible value is 100.
CORM = 100 * (sum of (j - i) * crm[i][j] for each i,j) / (N-1)
The CORM measure indicates the dominant lag of cross-recurrences.
Small CORM values indicate that the same fixations in both fixation sequences tend to occur close in time,
whereas large CORM values indicate that cross-recurrences tend to occur with either a large positive or negative lag.
"""


# calculates CORM
def calc_corm(crm):
    c = crm.sum()
    n = len(crm)
    acc = 0
    for i in range(n):
        for j in range(n):
            acc += (j - i) * crm[i][j]
    div = (n - 1) * c
    div = div if div != 0 else 1
    return 100 * (acc / div)


"""Utility functions"""


# calculates count and sum of lengths of all diagonal line longer then @param(length)
def calc_diagonal_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0

    for i in range(n):
        # get the ith diagonal (from main diagonal to right) of matrix
        diagonal = np.diag(matrix, i)
        temp_count, temp_count_length = calc_ones(diagonal, length)
        count += temp_count
        count_length += temp_count_length

        # calculates ith diagonal (from main diagonal to left) of matrix
        if i != 0:
            diagonal = np.diag(matrix, -i)
            temp_count, temp_count_length = calc_ones(diagonal, length)
            count += temp_count
            count_length += temp_count_length

    return count, count_length


# calculates count and sum of lengths of all horizontal line longer then @param(length)
def calc_horizontal_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0

    for i in range(n):
        horizontal = matrix[i]
        temp_count, temp_count_length = calc_ones(horizontal, length)
        count += temp_count
        count_length += temp_count_length

    return count, count_length


# calculates count and sum of lengths of all vertical line longer then @param(length)
def calc_vertical_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0

    for i in range(n):
        vertical = matrix[:, i]
        temp_count, temp_count_length = calc_ones(vertical, length)
        count += temp_count
        count_length += temp_count_length

    return count, count_length


# calculates count and sum of lengths of all 1's sequences in @param(vector) that are longer then @param(length)
def calc_ones(vector, length):
    index = 0
    count = 0
    count_length = 0
    while index < len(vector):
        l = 0
        while index < len(vector) and vector[index] == 1:
            l += 1
            index += 1
        index += 1
        if l >= length:
            count += 1
            count_length += l
    return count, count_length
