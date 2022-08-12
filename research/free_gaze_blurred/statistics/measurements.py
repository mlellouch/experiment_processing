# Import Python Libraries
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


def print_results():
    avg_results_by_blur, results, blinks_per_blur, saccades_per_blur = calc_results(span=2)
    labels = ["0.0-2.0", "2.0-4.0", "4.0-6.0", "6.0-8.0", "8.0-10.0", "10.0-12.0", "12.0-14.0", "14.0-16.0", "16.0-18.0", "18.0-20.0"]
    end_time = time.time()
    # print(f"results: {results}")

    # blinks per image blur level
    plt.figure()
    plt.scatter(blinks_per_blur.keys(), blinks_per_blur.values())
    plt.savefig('C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\blinks_per_blur.png')

    # saccades per image blur level
    plt.figure()
    plt.scatter(saccades_per_blur.keys(), saccades_per_blur.values())
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\saccades_per_blur.png')

    # linear distance heat map
    matrix1 = get_measurement_results_matrix("linear distance", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix1, linewidth=1, square=False, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\linear_distance.png')

    # cross recurrence
    matrix2 = get_measurement_results_matrix("cross recurrence", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix2, linewidth=1, square=True, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\cross_recurrence.png')

    # determinism
    matrix3 = get_measurement_results_matrix("determinism", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix3, linewidth=1, square=True, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\determinism.png')

    # laminarity
    matrix4 = get_measurement_results_matrix("laminarity", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix4, linewidth=1, square=True, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\laminarity.png')

    # center of recurrence mas
    matrix5 = get_measurement_results_matrix("CORM", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix5, linewidth=1, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\corm.png')

    # fixation overlap
    matrix6 = get_measurement_results_matrix("fixation overlap", avg_results_by_blur)
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(matrix6, linewidth=1, square=True, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.savefig(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\fixation_overlap.png')


def get_measurement_results_matrix(measurement, avg_results_by_blur):
    return [[avg_results_by_blur[i][j][1][measurement]
             for i in range(len(avg_results_by_blur))]
            for j in range(len(avg_results_by_blur))]


def calc_results(span=1):
    fixations_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\statistics\\fixations.csv')
    images_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\images.csv')
    blinks_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\blinks.csv')
    saccades_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\saccades.csv')
    blinks_per_blur = blinks_per_image_blur(blinks_df, images_df)
    saccades_per_blur = saccades_per_image_blur(saccades_df, images_df)
    # init scanpaths list
    fixations_sps_df = group_by_image(fixations_df)
    fixations_sps = [init_fixation_list(sp) for sp in fixations_sps_df]
    resampled_fixations_df = resample_fixations(images_df, fixations_df)
    resampled_fixations_sps_df = group_by_image(resampled_fixations_df)
    resampled_fixations_sps = [init_fixation_list_with_time(sp) for sp in resampled_fixations_sps_df]
    # init matrix of measurements results for each 2 scanpaths
    results = np.array([[{} for _ in range(len(fixations_sps))] for _ in range((len(fixations_sps)))])
    # init a matrix of measurements results average values for each 2 scanpaths of a specific blur range
    avg_results_by_blur = np.array([[(0, {"linear distance": 0,
                                          "cross recurrence": 0,
                                          "determinism": 0,
                                          "laminarity": 0,
                                          "CORM": 0,
                                          "fixation overlap": 0}) for _ in range(20 // span)] for _ in
                                    range(20 // span)])
    # calculate results and average results
    for i in range(len(fixations_sps)):
        for j in range(len(fixations_sps)):
            if i != j and len(fixations_sps[i]) > 0 and len(fixations_sps[j]) > 0:
                # TODO: figure out how to calculate distance with visual angle (1.9 and 3.5)
                crm = cross_recurrence_analysis(fixations_sps[i], fixations_sps[j], 100)
                overlap_percentage = fixation_overlap(resampled_fixations_sps[i], resampled_fixations_sps[j], 100)
                measures = {"linear distance": calc_linear_distance(fixations_sps[i], fixations_sps[j]),
                            "cross recurrence": calc_cross_recurrence(crm),
                            "determinism": calc_determinism(crm, 2, sec_diagonal=False),
                            "laminarity": calc_laminarity(crm, 2),
                            "CORM": calc_corm(crm),
                            "fixation overlap": overlap_percentage}
                results[i][j] = measures
                blur_i = images_df.loc[fixations_sps_df[i].loc[0, 'image_index'], 'blur']
                blur_j = images_df.loc[fixations_sps_df[j].loc[0, 'image_index'], 'blur']
                blur_idx1 = int(blur_i // span)
                blur_idx2 = int(blur_j // span)
                count, values = avg_results_by_blur[blur_idx1][blur_idx2]
                for key in measures:
                    value = measures[key]
                    values[key] = ((values[key] * count) + value) / (count + 1)
                avg_results_by_blur[blur_idx1][blur_idx2] = (count + 1, values)
    return avg_results_by_blur, results, blinks_per_blur, saccades_per_blur


def group_by_image(df):
    sps = []
    num = (1 + df['image_index'].max())
    for i in range(num):
        sp = df[df['image_index'] == i]
        sp.reset_index(drop=True, inplace=True)
        sps.append(sp)
    return sps


def saccades_per_image_blur(saccades_df, images_df):
    # add image_blur column to saccades_df
    for index, row in saccades_df.iterrows():
        saccades_df.loc[index, 'image_blur'] = images_df.loc[row['image_index'], 'blur']
    saccades_df = saccades_df.astype({"image_index": int}, errors='raise')
    # calc saccades per blur level
    saccades_per_blur = {}
    for index, row in saccades_df.iterrows():
        if row['image_blur'] in saccades_per_blur:
            saccades_per_blur[row['image_blur']] += 1
        else:
            saccades_per_blur[row['image_blur']] = 0
    # normalize 0.0 blur level
    zero_blur_count = 0
    for index, row in images_df.iterrows():
        if row['blur'] == 0.0:
            zero_blur_count += 1
    saccades_per_blur[0.0] = saccades_per_blur[0.0] / zero_blur_count
    return saccades_per_blur
    plt.scatter(saccades_per_blur.keys(), saccades_per_blur.values())
    plt.show()


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
    return blinks_per_blur


def fixation_overlap(fixations1, fixations2, radius):
    # make equal length
    fixations1_copy = fixations1.copy()
    fixations2_copy = fixations2.copy()
    # truncate the longer list
    del fixations1_copy[len(fixations2_copy):]
    del fixations2_copy[len(fixations1_copy):]
    count_overlaps = 0
    # calculates list of distances for each pair of equal index entries
    for i in range(len(fixations1_copy)):
        if overlaps(fixations1_copy[i][0], fixations1_copy[i][1], fixations2_copy[i][0], fixations2_copy[i][1], radius):
            count_overlaps += 1
    length = len(fixations1_copy) if len(fixations1_copy) > 0 else 1
    return 100 * (count_overlaps / length)


def calc_linear_distance(fixations1, fixations2, normalize=True):
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


def calc_laminarity(crm, length, by_length=True):
    count, count_length = calc_vertical_lines(crm, length)
    count2, count_length2 = calc_diagonal_lines(crm, length)
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


def calc_vertical_lines(matrix, length):
    n = len(matrix)
    count = 0
    count_length = 0
    for i in range(n):
        horizontal = matrix[:, i]
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


def cross_recurrence_analysis(fixations1, fixations2, radius):
    fixations1_copy = fixations1.copy()
    fixations2_copy = fixations2.copy()
    # truncate the longer list
    del fixations1_copy[len(fixations2_copy):]
    del fixations2_copy[len(fixations1_copy):]

    distances = cdist(fixations1_copy, fixations2_copy)
    crm = np.zeros((len(fixations1_copy), len(fixations2_copy)))
    for i in range(len(fixations1_copy)):
        for j in range(len(fixations2_copy)):
            crm[i][j] = 1 if distances[i][j] <= radius else 0
    return crm


def init_fixation_list(fixation_df):
    fixations = []
    for index, row in fixation_df.iterrows():
        point = np.array((fixation_df.loc[index, 'image_x'], fixation_df.loc[index, 'image_y']))
        fixations.append(point)
    return fixations


def init_fixation_list_with_time(fixation_df):
    fixations = []
    for index, row in fixation_df.iterrows():
        point = np.array((fixation_df.loc[index, 'image_x'], fixation_df.loc[index, 'image_y']))
        interval = pd.Interval(fixation_df.loc[index, 'start time'], fixation_df.loc[index, 'end time'])
        fixations.append([point, interval])
    return fixations


def resample_fixations(images_df, fixations_df):
    ret_df = fixations_df.copy()
    for index, _ in ret_df.iterrows():
        fixation_start_time = fixations_df.loc[index, 'start time']
        image_start_time = images_df.loc[ret_df.loc[index, 'image_index'], 'start time']
        ret_df.loc[index, 'start time'] = fixation_start_time - image_start_time
    return ret_df


def overlaps(point1, interval1, point2, interval2, radius):
    return interval1.overlaps(interval2) and calc_dist(point1, point2) <= radius


def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


if __name__ == '__main__':
    print_results()
