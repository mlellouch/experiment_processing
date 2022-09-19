import os
import seaborn as sns
import numpy as np
import pandas as pd
from fixation_overlap import fixation_overlap
from cross_recurrence_quantification_analysis import cross_recurrence_analysis
from linear_distance import calc_linear_distance
from cross_recurrence_quantification_analysis import calc_cross_recurrence
from research.free_gaze_blurred.statistics.scanpath_comparison.attribute_by_image_blur import attribute_per_image_blur
from research.free_gaze_blurred.statistics.scanpath_comparison.cross_recurrence_quantification_analysis import calc_cross_recurrence, \
    cross_recurrence_analysis, calc_determinism, calc_laminarity, calc_corm
from research.free_gaze_blurred.statistics.scanpath_comparison.fixation_overlap import fixation_overlap, resample_fixations
from research.free_gaze_blurred.statistics.scanpath_comparison.linear_distance import calc_linear_distance
import matplotlib.pyplot as plt

def get_matrices(path1, path2, radius=50):

    fixations1_df = pd.read_csv(os.path.join(path1, 'fixations.csv'))
    images1_df = pd.read_csv(os.path.join(path1, 'images.csv'))

    fixations2_df = pd.read_csv(os.path.join(path2, 'fixations.csv'))
    images2_df = pd.read_csv(os.path.join(path2, 'images.csv'))

    blurs = pd.unique(images1_df['blur']).tolist()
    blurs.sort()
    blurs.remove(0)
    mat_size = (len(blurs), len(blurs))

    linear_distance_mat = np.zeros(mat_size, dtype=float)
    corm_mat = np.zeros(mat_size, dtype=float)
    cross_recurrence_mat = np.zeros(mat_size, dtype=float)
    determinism_mat = np.zeros(mat_size, dtype=float)
    laminarity_mat = np.zeros(mat_size, dtype=float)
    count_mat = np.zeros(mat_size, dtype=float) + 1e-7

    def path_to_image_name(full_name):
        return full_name.split('_')[0]

    for idx1, img1 in images1_df.iterrows():
        if idx1 == 0:
            continue
        image_name1 = path_to_image_name(img1['path'])
        blur1 = img1['blur']
        if blur1 == 0.0:
            continue
        for idx2, img2 in images2_df.iterrows():
            image_name2 = path_to_image_name(img2['path'])
            if image_name1 == image_name2:
                blur2 = img2['blur']

                # crm - cross recurrence matrix
                fixation1 = fixations1_df[fixations1_df['image_index'] == idx1]
                fixation2 = fixations2_df[fixations2_df['image_index'] == idx2]
                crm = cross_recurrence_analysis(fixation1, fixation2, radius)

                # calculates all measurement methods and assign
                measures = {"linear distance": calc_linear_distance(fixation1, fixation2),
                            "cross recurrence": calc_cross_recurrence(crm),
                            "determinism": calc_determinism(crm, 2, sec_diagonal=False),
                            "laminarity": calc_laminarity(crm, 2),
                            "CORM": calc_corm(crm),
                }
                i, j = blurs.index(blur1), blurs.index(blur2)
                count_mat[i, j] += 1
                linear_distance_mat[i, j] += measures['linear distance']
                cross_recurrence_mat[i, j] += measures['cross recurrence']
                determinism_mat[i, j] += measures['determinism']
                laminarity_mat[i, j] += measures['laminarity']
                corm_mat[i, j] += measures['CORM']

                count_mat[j, i] += 1
                linear_distance_mat[j, i] += measures['linear distance']
                cross_recurrence_mat[j, i] += measures['cross recurrence']
                determinism_mat[j, i] += measures['determinism']
                laminarity_mat[j, i] += measures['laminarity']
                corm_mat[j, i] += measures['CORM']

    linear_distance_mat /= count_mat
    cross_recurrence_mat /= count_mat
    determinism_mat /= count_mat
    laminarity_mat /= count_mat
    corm_mat /= count_mat
    return linear_distance_mat, cross_recurrence_mat, determinism_mat, laminarity_mat, corm_mat, blurs


def run(input_path1, input_path2, output_path, radius):
    linear_distance, cross_recurrence, determinsm, laminaty, corm, blurs = get_matrices(input_path1, input_path2, radius)

    def save_mat(mat, path, title):
        ax = sns.heatmap(mat, linewidth=1, square=False, annot=True, fmt=".1f", xticklabels=blurs,
                         yticklabels=blurs)
        ax.set_title(title)
        plt.savefig(path)
        plt.clf()

    graphs = [
        {'mat': linear_distance, 'name': 'linear_distance', 'title': 'linear distance'},
        {'mat': cross_recurrence, 'name': 'cross_recurrence', 'title': 'cross recurrence'},
        {'mat': determinsm, 'name': 'determinism', 'title': 'determinism'},
        {'mat': laminaty, 'name': 'laminarity', 'title': 'laminarity'},
        {'mat': corm, 'name': 'corm', 'title': 'corm'}
    ]

    for g in graphs:
        save_mat(g['mat'], os.path.join(output_path, g['name']), g['title'])



if __name__ == '__main__':
    pair = '../../../../outputs/preprocessed_outputs/FGBS/pair2'
    path1 = os.path.join(pair, 'first')
    path2 = os.path.join(pair, 'second')
    output_path = '../outputs/pair_comparison2'
    run(path1, path2, output_path, radius=100)
