# Import Python Libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.spatial.distance import cdist


def print_results():
    crm = cross_recurrence_analysis("fixations_test.csv", "fixations_test.csv", 50)
    print(f"crm: {crm}")
    rec = calc_cross_recurrence(crm)
    print(f"REC = {rec}")
    #start_time = time.time()
    det = calc_determinism(crm, 2, by_length=False, sec_diagonal=True)
    #end_time = time.time()
    print(f"DET = {det}")
    #time_lapsed = end_time - start_time
    #time_convert(time_lapsed)


def fixation_overlap(sample_df1, sample_df2, radius):
    # make equal length
    diff = sample_df1.shape[0] - sample_df2.shape[0]
    if diff > 0:
        sample_df1 = sample_df1.loc[0: -diff]
    elif diff < 0:
        sample_df2 = sample_df2.loc[0: diff]
    distances = {}
    sum = 0
    count = 0
    # calculates list of distances for each pair of equal index entries
    for index in range(sample_df1.shape[0]):
        if sample_df1.loc[index, 'is_fixation'] and sample_df2.loc[index, 'is_fixation']:
            point1 = np.array((sample_df1.loc[index, 'image_x'], sample_df1.loc[index, 'image_y']))
            point2 = np.array((sample_df2.loc[index, 'image_x'], sample_df2.loc[index, 'image_y']))
            distances[index] = calc_dist(point1, point2)
            sum += distances[index]
            count +=1
        else:
            distances[index] = None
    similarity = sum/count
    return similarity , distances


def calc_cross_recurrence(crm):
    n = len(crm)
    c = crm.sum()
    return 100 * (c / (n ** 2))


def calc_determinism(crm, length, by_length=True, sec_diagonal=True):
    count, count_length = calc_diagonal_lines(crm, length)
    if sec_diagonal:
        flipped_crm = np.fliplr(crm)
        print(f"flipped: \n{flipped_crm}")
        count2, count_length2 = calc_diagonal_lines(flipped_crm, length)
        count += count2
        count_length += count_length2
    return count_length if by_length else count


def calc_diagonal_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0
    for i in range(n):
        diag = np.diag(matrix, i)
        temp_count, temp_count_length = calc_ones(diag, length)
        count += temp_count
        count_length += temp_count_length
        if i != 0:
            diag = np.diag(matrix, -i)
            temp_count, temp_count_length = calc_ones(diag, length)
            count += temp_count
            count_length += temp_count_length
    return count, count_length


def calc_ones(diag, length):
    index = 0
    count = 0
    count_length = 0

    while index < len(diag):
        l = 0
        while index < len(diag) and diag[index] == 1:
            l += 1
            index += 1
        index += 1
        if l >= length:
            count += 1
            count_length += l
    return count, count_length


def cross_recurrence_analysis(sample_file1, sample_file2, radius):
    sample_df1 = pd.read_csv(sample_file1)
    sample_df2 = pd.read_csv(sample_file2)
    fixations1 = init_fixation_list(sample_df1)
    fixations2 = init_fixation_list(sample_df2)

    # truncate the longer list
    del fixations1[len(fixations2):]
    del fixations2[len(fixations1):]

    distances = cdist(fixations1, fixations2)
    crm = np.zeros((len(fixations1), len(fixations2)))
    for i in range(len(fixations1)):
        for j in range(len(fixations2)):
            crm[i][j] = 1 if distances[i][j] <= radius else 0

    return crm


def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def init_fixation_list(fixation_df):
    fixations = []
    for index in range(fixation_df.shape[0]):
        point = np.array((fixation_df.loc[index, 'x'], fixation_df.loc[index, 'y']))
        fixations.append(point)
    return fixations


if __name__ == '__main__':
    print(fixation_overlap(pd.read_csv("sample_test.csv"), pd.read_csv("sample_test.csv"), 10))
    # blinks_per_image()
