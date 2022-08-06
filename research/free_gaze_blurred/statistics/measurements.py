# Import Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.spatial.distance import cdist


def print_results():
    crm = cross_recurrence_analysis("fixations_test.csv", "fixations_test.csv", 50)
    print(f"crm: {crm}")
    rec = calc_cross_recurrence(crm)
    print(f"REC = {rec}")
    det = calc_determinism(crm, 2, by_length=False, sec_diagonal=True)
    print(f"DET = {det}")


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
    # calculates list of distances for each pair of equal index entries
    for index in range(sample_df1.shape[0]):
        if sample_df1.loc[index, 'is_fixation'] and sample_df2.loc[index, 'is_fixation']:
            point1 = np.array((sample_df1.loc[index, 'image_x'], sample_df1.loc[index, 'image_y']))
            point2 = np.array((sample_df2.loc[index, 'image_x'], sample_df2.loc[index, 'image_y']))
            distances[index] = calc_dist(point1, point2)
            dist_sum += distances[index]
            count +=1
        else:
            distances[index] = None
    similarity = dist_sum / count
    return similarity, distances


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
            acc += (j-i) * crm[i][j]
    return 100 * (acc / (n-1)*c)


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


def init_fixation_list(fixation_df):
    fixations = []
    for index in range(fixation_df.shape[0]):
        point = np.array((fixation_df.loc[index, 'x'], fixation_df.loc[index, 'y']))
        fixations.append(point)
    return fixations


def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def testing():
    matrix = np.array([[1,2,3],
                      [4,5,6]])
    print(len(matrix))


if __name__ == '__main__':
    testing()
    # blinks_per_image()
    #
