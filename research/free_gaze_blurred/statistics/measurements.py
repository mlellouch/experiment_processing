# Import Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist


def print_results():
    avg_results_by_blur, results = calc_results()
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


def get_measurement_results_matrix(measurement, avg_results_by_blur):
    return [[avg_results_by_blur[i][j][1][measurement]
            for i in range(len(avg_results_by_blur))]
            for j in range(len(avg_results_by_blur))]


def calc_results():
    df = pd.read_csv('fixations.csv')
    images_df = pd.read_csv('images.csv')
    # init scanpaths list
    sps = group_by_image(df)
    # init matrix of measurements results for each 2 scanpaths
    results = np.zeros(len(sps), len(sps))
    # init a matrix of measurements results average values for each 2 scanpaths of a specific blur range
    avg_results_by_blur = np.array([[(0, {"linear distance": 0,
                                          "cross recurrence": 0,
                                          "determinism": 0,
                                          "laminarity": 0,
                                          "CORM": 0,
                                          "fixation overlap": 0}) for _ in range(20)] for _ in range(20)])
    # calculate results and average results
    for i in range(len(sps)):
        for j in range(len(sps)):
            if i != j:
                # TODO: figure out how to calculate distance with visual angle (1.9 and 3.5)
                crm = cross_recurrence_analysis(sps[i], sps[j], 1.9)
                overlap_percentage, _, _ = fixation_overlap(sps[i], sps[j], 3.5)
                measures = {"linear distance": calc_linear_distance(sps[i], sps[j]),
                            "cross recurrence": calc_cross_recurrence(crm),
                            "determinism": calc_determinism(crm, 2),
                            "laminarity": calc_laminarity(crm, 2),
                            "CORM": calc_corm(crm),
                            "fixation overlap": overlap_percentage}
                results[i][j] = measures
                blur_idx1 = int(images_df.loc[sps[i].loc[0, 'image_index'], 'blur level'])
                blur_idx2 = int(images_df.loc[sps[j].loc[0, 'image_index'], 'blur level'])
                count, values = avg_results_by_blur[blur_idx1][blur_idx2]
                for key, value in measures:
                    values[key] = ((values[key] * count) + value) / (count + 1)
                avg_results_by_blur[i][j] = (count + 1, values)
    return avg_results_by_blur, results


def group_by_image(df):
    sps = []
    num = (1 + df['image_index'].max())
    for i in range(num):
        sps.append(df[df['image_index'] == i])
    return sps


def blinks_per_image_blur(blinks_df, images_df):
    # add blur_level column to images_df
    for index, image_row in images_df.iterrows():
        path = image_row['path']
        if len(path.split('_')) > 1:
            images_df.loc[index, 'blur_level'] = path.split('_')[1][0: -4]
        else:
            images_df.loc[index, 'blur_level'] = 0
    images_df = images_df.astype({"blur_level": float}, errors='raise')
    # add image_blur column to blinks_df
    for index, row in blinks_df.iterrows():
        blinks_df.loc[index, 'image_blur'] = images_df.loc[row['image_index'], 'blur_level']
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
        if row['blur_level'] == 0.0:
            zero_blur_count += 1
    blinks_per_blur[0.0] = blinks_per_blur[0.0] / zero_blur_count
    # scatter plot results
    plt.scatter(blinks_per_blur.keys(), blinks_per_blur.values())
    plt.show()


def fixation_overlap(sample_df1, sample_df2, radius):
    # make equal length
    diff = sample_df1.shape[0] - sample_df2.shape[0]
    if diff > 0:
        sample_df1 = sample_df1.loc[0: -diff]
    elif diff < 0:
        sample_df2 = sample_df2.loc[0: diff]
    distances = {}
    dist_sum = 0
    count = 0
    overlaps = 0
    # calculates list of distances for each pair of equal index entries
    for index in range(sample_df1.shape[0]):
        if sample_df1.loc[index, 'is_fixation'] and sample_df2.loc[index, 'is_fixation']:
            point1 = np.array((sample_df1.loc[index, 'image_x'], sample_df1.loc[index, 'image_y']))
            point2 = np.array((sample_df2.loc[index, 'image_x'], sample_df2.loc[index, 'image_y']))
            distances[index] = calc_dist(point1, point2)
            if distances[index] <= radius:
                overlaps += 1
            dist_sum += distances[index]
            count += 1
        else:
            distances[index] = None
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
    return 100 * (count_length / c) if by_length else 100 * (count / c)


def calc_corm(crm):
    c = crm.sum()
    n = len(crm)
    acc = 0
    for i in range(n):
        for j in range(n):
            acc += (j - i) * crm[i][j]
    return 100 * (acc / (n - 1) * c)


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
        horizontal = matrix[1]
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
    for index in range(fixation_df.shape[0]):
        point = np.array((fixation_df.loc[index, 'x'], fixation_df.loc[index, 'y']))
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
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6]])
    print(int(1.9))


if __name__ == '__main__':
    testing()
    # blinks_per_image()
    #
