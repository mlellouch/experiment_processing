import numpy as np

"""
Fixation overlap
defined as the proportion of overlapping samples.
Two samples (at time t) overlap if the Euclidean distance between two
samples is less than a predefined threshold.
the scanpaths are first resampled uniformly in time and truncated to the shorter length.
"""


def fixation_overlap(fixations1, fixations2, radius):
    # make equal length
    fixations1_copy = fixations1.copy()
    fixations2_copy = fixations2.copy()

    # truncate the longer list
    del fixations1_copy[len(fixations2_copy):]
    del fixations2_copy[len(fixations1_copy):]

    # calculates list of distances for each pair of equal index entries
    count_overlaps = 0
    for i in range(len(fixations1_copy)):
        if overlaps(fixations1_copy[i][0], fixations1_copy[i][1], fixations2_copy[i][0], fixations2_copy[i][1], radius):
            count_overlaps += 1
    length = len(fixations1_copy) if len(fixations1_copy) > 0 else 1

    return 100 * (count_overlaps / length)


# assesses whether or not two fixations overlap
def overlaps(point1, interval1, point2, interval2, radius):
    return interval1.overlaps(interval2) and calc_dist(point1, point2) <= radius


# calculate the euclidean distance between to points (fixations)
def calc_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


# return a fixations data frame where the start and end time of each fixation is relative to the image start time
def resample_fixations(images_df, fixations_df):
    ret_df = fixations_df.copy()
    for index, _ in ret_df.iterrows():
        fixation_start_time = fixations_df.loc[index, 'start time']
        fixation_end_time = fixations_df.loc[index, 'end time']
        image_start_time = images_df.loc[ret_df.loc[index, 'image_index'], 'start time']
        ret_df.loc[index, 'start time'] = fixation_start_time - image_start_time
        ret_df.loc[index, 'end time'] = fixation_end_time - image_start_time
    return ret_df
