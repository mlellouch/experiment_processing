# Import Python Libraries
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


def print_results():
    start_time = time.time()
    avg_results_by_blur, results = calc_results()
    end_time = time.time()
    time_convert(end_time - start_time)
    print(f"results: {results}")
    print(f"avg_results_by_blur: {avg_results_by_blur}")

    # linear distance heat map
    matrix = get_measurement_results_matrix("linear distance", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()

    # cross recurrence
    matrix = get_measurement_results_matrix("cross recurrence", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()

    # determinism
    matrix = get_measurement_results_matrix("determinism", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()

    # laminarity
    matrix = get_measurement_results_matrix("laminarity", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()

    # center of recurrence mas
    matrix = get_measurement_results_matrix("CORM", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()

    # fixation overlap
    matrix = get_measurement_results_matrix("fixation overlap", avg_results_by_blur)
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.show()


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}:{1}:{2}".format(int(hours), int(mins), sec))


def get_measurement_results_matrix(measurement, avg_results_by_blur):
    return [[avg_results_by_blur[i][j][1][measurement]
            for i in range(len(avg_results_by_blur))]
            for j in range(len(avg_results_by_blur))]


def calc_results():
    fixations_df = pd.read_csv('C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\statistics\\fixations.csv')
    images_df = pd.read_csv('C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\images.csv')
    samples_df = pd.read_csv('C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\samples.csv')
    # init scanpaths list
    fixations_sps = group_by_image(fixations_df)
    samples_sps = group_by_image(samples_df)
    # init matrix of measurements results for each 2 scanpaths
    results = np.array([[{} for _ in range(len(fixations_sps))] for _ in range((len(fixations_sps)))])
    # init a matrix of measurements results average values for each 2 scanpaths of a specific blur range
    avg_results_by_blur = np.array([[(0, {"linear distance": 0,
                                          "cross recurrence": 0,
                                          "determinism": 0,
                                          "laminarity": 0,
                                          "CORM": 0,
                                          "fixation overlap": 0}) for _ in range(20)] for _ in range(20)])
    # calculate results and average results
    for i in range(len(fixations_sps)):
        for j in range(len(fixations_sps)):
            if i != j and fixations_sps[i].shape[0] > 0 and fixations_sps[j].shape[0] > 0:
                # TODO: figure out how to calculate distance with visual angle (1.9 and 3.5)
                crm = cross_recurrence_analysis(fixations_sps[i], fixations_sps[j], 1.9)
                overlap_percentage, _, _ = fixation_overlap(samples_sps[i], samples_sps[j], 3.5)
                measures = {"linear distance": calc_linear_distance(fixations_sps[i], fixations_sps[j]),
                            "cross recurrence": calc_cross_recurrence(crm),
                            "determinism": calc_determinism(crm, 2),
                            "laminarity": calc_laminarity(crm, 2),
                            "CORM": calc_corm(crm),
                            "fixation overlap": overlap_percentage}
                results[i][j] = measures
                blur_idx1 = int(images_df.loc[fixations_sps[i].loc[0, 'image_index'], 'blur'])
                blur_idx2 = int(images_df.loc[fixations_sps[j].loc[0, 'image_index'], 'blur'])
                count, values = avg_results_by_blur[blur_idx1][blur_idx2]
                for key in measures:
                    value = measures[key]
                    values[key] = ((values[key] * count) + value) / (count + 1)
                avg_results_by_blur[blur_idx1][blur_idx2] = (count + 1, values)
    return avg_results_by_blur, results


def group_by_image(df):
    sps = []
    num = (1 + df['image_index'].max())
    for i in range(num):
        sp = df[df['image_index'] == i]
        sp.reset_index(drop=True, inplace=True)
        sps.append(sp)
    return sps


def blinks_per_image_blur(blinks_df, images_df):
    # add image_blur column to blinks_df
    for index, row in blinks_df.iterrows():
        blinks_df.loc[index, 'image_blur'] = images_df.loc[row['image_index'], 'blur']
    blinks_df = blinks_df.astype({"image_index": int}, errors='raise')
    # calc blinks per blur level
    blinks_per_blur = {}
    for index, row in blinks_df.iterrows():
        if row['image_blur'] in blinks_per_blur:
            blinks_per_blur[row['image_blur']] += 1
        else:
            blinks_per_blur[row['image_blur']] = 0
    # normalize 0.0 blur level
    zero_blur_count = 0
    for index, row in images_df.iterrows():
        if row['blur'] == 0.0:
            zero_blur_count += 1
    blinks_per_blur[0.0] = blinks_per_blur[0.0] / zero_blur_count
    # scatter plot results
    plt.scatter(blinks_per_blur.keys(), blinks_per_blur.values())
    plt.show()


def fixation_overlap(sample_df1, sample_df2, radius):
    # make equal length
    diff = sample_df1.shape[0] - sample_df2.shape[0]
    if diff > 0:
        sample_df1 = sample_df1.loc[0: sample_df2.shape[0]-1]
    elif diff < 0:
        sample_df2 = sample_df2.loc[0: sample_df1.shape[0]-1]
    distances = {}
    dist_sum = 0
    count = 0
    overlaps = 0
    # calculates list of distances for each pair of equal index entries
    for index1, row in sample_df1.iterrows():
        if sample_df1.loc[index1, 'is_fixation'] and sample_df2.loc[index1, 'is_fixation']:
            point1 = np.array((sample_df1.loc[index1, 'image_x'], sample_df1.loc[index1, 'image_y']))
            point2 = np.array((sample_df2.loc[index1, 'image_x'], sample_df2.loc[index1, 'image_y']))
            distances[index1] = calc_dist(point1, point2)
            if distances[index1] <= radius:
                overlaps += 1
            dist_sum += distances[index1]
            count += 1
        else:
            distances[index1] = None
    count = count if count != 0 else 1
    similarity = dist_sum / count
    percentage = overlaps / count
    return percentage, similarity, distances


def calc_linear_distance(fixations_df1, fixations_df2, normalize=True):
    fixations1 = init_fixation_list(fixations_df1)
    fixations2 = init_fixation_list(fixations_df2)
    distances = cdist(fixations1, fixations2)
    d = distances.min(axis=0).sum() + distances.min(axis=1).sum()
    if normalize:
        d /= max(len(fixations1), len(fixations2))
    return d


def calc_cross_recurrence(crm):
    n = len(crm)
    c = crm.sum()
    return 100 * (c / (n ** 2))


def calc_determinism(crm, length, by_length=True, sec_diagonal=True):
    count, count_length = calc_diagonal_lines(crm, length)
    if sec_diagonal:
        flipped_crm = np.fliplr(crm)
        count2, count_length2 = calc_diagonal_lines(flipped_crm, length)
        count += count2
        count_length += count_length2
    c = crm.sum()
    c = c if c != 0 else 1
    return 100 * (count_length / c) if by_length else 100 * (count / c)


def calc_laminarity(crm, length, by_length=True, sec_diagonal=True):
    count, count_length = calc_horizontal_lines(crm, length)
    count2, count_length2 = calc_diagonal_lines(crm, length)
    count += count2
    count_length += count_length2
    if sec_diagonal:
        flipped_crm = np.fliplr(crm)
        count2, count_length2 = calc_diagonal_lines(flipped_crm, length)
        count += count2
        count_length += count_length2
    c = crm.sum()
    c = c if c != 0 else 1
    return 100 * (count_length / c) if by_length else 100 * (count / c)


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


def calc_diagonal_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0
    for i in range(n):
        diagonal = np.diag(matrix, i)
        temp_count, temp_count_length = calc_ones(diagonal, length)
        count += temp_count
        count_length += temp_count_length
        if i != 0:
            diagonal = np.diag(matrix, -i)
            temp_count, temp_count_length = calc_ones(diagonal, length)
            count += temp_count
            count_length += temp_count_length
    return count, count_length


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


def cross_recurrence_analysis(sample_df1, sample_df2, radius):
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


def init_fixation_list(fixation_df):
    fixations = []
    for index, row in fixation_df.iterrows():
        point = np.array((fixation_df.loc[index, 'image_x'], fixation_df.loc[index, 'image_y']))
        fixations.append(point)
    return fixations


def resample_fixations(images_df, fixations_df):
    ret_df = fixations_df.copy()
    for index, _ in ret_df.iterrows():
        fixation_start_time = fixations_df.loc[index, 'start time']
        image_start_time = images_df.loc[ret_df.loc[index, 'image_index'], 'start time']
        ret_df.loc[index, 'start time'] = fixation_start_time - image_start_time
    return ret_df


def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def testing():
    matrix = np.zeros(181, 181)
    print(int(1.9))


if __name__ == '__main__':
    # testing()
    print_results()
    # blinks_per_image()
    #
