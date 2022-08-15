# Import Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from research.free_gaze_blurred.statistics.attribute_by_image_blur import attribute_per_image_blur
from research.free_gaze_blurred.statistics.cross_recurrence_quantification_analysis import calc_cross_recurrence, \
    cross_recurrence_analysis, calc_determinism, calc_laminarity, calc_corm
from research.free_gaze_blurred.statistics.fixation_overlap import fixation_overlap, resample_fixations
from research.free_gaze_blurred.statistics.linear_distance import calc_linear_distance


# calls the main calculation method and visualizes the results
def print_results():

    avg_results_by_blur, results, blinks_per_blur, saccades_per_blur = calc_results(span=2)

    labels = ["0.0-2.0", "2.0-4.0", "4.0-6.0", "6.0-8.0", "8.0-10.0", "10.0-12.0", "12.0-14.0", "14.0-16.0", "16.0-18.0", "18.0-20.0"]

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


# main calculation method, parameter span determines the span of each blur range (i.e 0-1 1-2... or 0-5, 5-10,...)
def calc_results(span=1):

    # reads data frames
    fixations_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\statistics\\fixations.csv')
    images_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\images.csv')
    blinks_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\blinks.csv')
    saccades_df = pd.read_csv(
        'C:\\Users\\levil\\OneDrive\\שולחן העבודה\\experiment_processing\\experiment_processing\\research\\free_gaze_blurred\\statistics\\saccades.csv')

    # calculates blinks and saccades per image blur level
    blinks_per_blur = attribute_per_image_blur(blinks_df, images_df)
    saccades_per_blur = attribute_per_image_blur(saccades_df, images_df)

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

                # crm - cross recurrence matrix
                crm = cross_recurrence_analysis(fixations_sps[i], fixations_sps[j], 100)

                # overlap similarity
                overlap_percentage = fixation_overlap(resampled_fixations_sps[i], resampled_fixations_sps[j], 100)

                # calculates all measurement methods and assign
                measures = {"linear distance": calc_linear_distance(fixations_sps[i], fixations_sps[j]),
                            "cross recurrence": calc_cross_recurrence(crm),
                            "determinism": calc_determinism(crm, 2, sec_diagonal=False),
                            "laminarity": calc_laminarity(crm, 2),
                            "CORM": calc_corm(crm),
                            "fixation overlap": overlap_percentage}
                results[i][j] = measures

                # calculates index of blur range, given the span
                blur_i = images_df.loc[fixations_sps_df[i].loc[0, 'image_index'], 'blur']
                blur_j = images_df.loc[fixations_sps_df[j].loc[0, 'image_index'], 'blur']
                blur_idx1 = int(blur_i // span)
                blur_idx2 = int(blur_j // span)

                # recalculates the averages
                count, values = avg_results_by_blur[blur_idx1][blur_idx2]
                for key in measures:
                    value = measures[key]
                    values[key] = ((values[key] * count) + value) / (count + 1)
                avg_results_by_blur[blur_idx1][blur_idx2] = (count + 1, values)

    return avg_results_by_blur, results, blinks_per_blur, saccades_per_blur


def get_measurement_results_matrix(measurement, avg_results_by_blur):
    return [[avg_results_by_blur[i][j][1][measurement]
             for i in range(len(avg_results_by_blur))]
            for j in range(len(avg_results_by_blur))]


def group_by_image(df):
    sps = []
    num = (1 + df['image_index'].max())
    for i in range(num):
        sp = df[df['image_index'] == i]
        sp.reset_index(drop=True, inplace=True)
        sps.append(sp)
    return sps


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


if __name__ == '__main__':
    print_results()
