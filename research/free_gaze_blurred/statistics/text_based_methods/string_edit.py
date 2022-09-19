import research.transformer.generate_text as generate_text
import pandas as pd
import numpy as np
import os
from argparse import Namespace
import seaborn as sns
from tqdm import tqdm
from itertools import product
import editdistance
import seaborn as sns
import matplotlib.pyplot as plt


def compare_scanpaths(samples1, samples2, cell_size=16, image_size=1024, subsample:int=4):
    args = {
        'new_line': False,
        'fixations_only': False,
        'skip_saccade_samples': False,
        'special_saccade_encode': False,
        'encode_time': False,
        'cell_size': cell_size,
        'image_size': image_size,
        'cell_based_location': True
    }
    args = Namespace(**args)

    samples1 = samples1[samples1.index % subsample == 0]
    samples2 = samples2[samples2.index % subsample == 0]

    text1 = generate_text.samples_to_text(args, samples1)
    text2 = generate_text.samples_to_text(args, samples2)



    val = editdistance.eval(text1, text2)
    return val


def get_edit_distance_matrix(image_texts):
    blurs = list(image_texts.keys())
    blurs.sort()

    mat = np.zeros([len(blurs), len(blurs)], dtype=float)

    for i in tqdm(range(len(blurs))):
        for j in range(i, len(blurs)):
            total = 0
            count = 0
            first = blurs[i]
            second = blurs[j]
            for a,b in product(first, second):
                if a == b:
                    continue
                total += editdistance.eval(a,b)
                count += 1

            mat[i, j] = total / count
            mat[j, i] = total / count

    return mat


def run(source, dest, subsample:int=4, only_for_same_images:bool=False):
    samples = pd.read_csv(os.path.join(source, 'samples.csv'))
    images = pd.read_csv(os.path.join(source, 'images.csv'))
    # arguments for a simple string
    args = {
        'new_line': False,
        'fixations_only': False,
        'skip_saccade_samples': False,
        'special_saccade_encode': False,
        'encode_time': False,
        'cell_size': 16,
        'image_size': 1024,
        'cell_based_location': True
    }
    args = Namespace(**args)

    image_texts = {}
    for image_index in tqdm(range(1, len(images))):
        image_blur = images.iloc[image_index]['blur']
        if image_blur == 0:
            continue
        if image_blur not in image_texts.keys():
            image_texts[image_blur] = []

        image_samples = samples.loc[samples['image_index'] == image_index]
        image_text = generate_text.samples_to_text(args, image_samples)
        image_texts[image_blur].append(image_text)

    final_mat = get_edit_distance_matrix(image_texts)

    ax = sns.heatmap(final_mat, linewidth=1, square=True, annot=True, fmt=".1f", xticklabels=image_texts.keys(), yticklabels=image_texts.keys())
    ax.set_title('Edit Distance')
    plt.savefig(os.path.join(dest, 'editdistance.png'))



def get_pair_matrix(samples1_df, images1_df, samples2_df, images2_df, cell_size=16, image_size=1024, subsample=4):
    # build mat
    blurs = pd.unique(images1_df['blur']).tolist()
    blurs.sort()
    blurs.remove(0)
    mat_size = (len(blurs), len(blurs))

    edit_distance_mat = np.zeros(mat_size, dtype=int)
    count_mat = np.zeros(mat_size, dtype=float) + 1e-7

    def path_to_image_name(full_name):
        return full_name.split('_')[0]

    for idx1, img1 in tqdm(images1_df.iterrows(), total=len(images1_df)):
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

                edit_distance = compare_scanpaths(
                    samples1_df[samples1_df['image_index'] == idx1],
                    samples2_df[samples2_df['image_index'] == idx2],
                    cell_size=cell_size,
                    image_size=image_size,
                    subsample=subsample
                )

                i, j = blurs.index(blur1), blurs.index(blur2)
                edit_distance_mat[i, j] += edit_distance
                edit_distance_mat[j, i] += edit_distance
                count_mat[i, j] += 1
                count_mat[j, i] += 1

    return edit_distance_mat / count_mat


def run_on_pair(path1, path2, output, cell_size=16, image_size=1024, subsample=4):
    samples1_df, images1_df = pd.read_csv(os.path.join(path1, 'samples.csv')), pd.read_csv(os.path.join(path1, 'images.csv'))
    samples2_df, images2_df = pd.read_csv(os.path.join(path2, 'samples.csv')), pd.read_csv(os.path.join(path2, 'images.csv'))

    # build mat
    blurs = pd.unique(images1_df['blur']).tolist()
    blurs.sort()
    blurs.remove(0)
    mat_size = (len(blurs), len(blurs))

    mat = get_pair_matrix(samples1_df, images1_df, samples2_df, images2_df, cell_size=cell_size, image_size=image_size, subsample=subsample)
    ax = sns.heatmap(mat, linewidth=1, square=False, annot=True, fmt=".1f", xticklabels=blurs,
                     yticklabels=blurs)
    ax.set_title('edit distance')
    plt.savefig(os.path.join(output, f'edit_distance_{cell_size}_{subsample}.png'))
    plt.clf()





if __name__ == '__main__':
    # source = '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur'
    # dest = '../outputs/scenes_aligned/'
    # run(source, dest, only_for_same_images=False)

    pair = '../../../../outputs/preprocessed_outputs/FGBS/pair2'
    path1 = os.path.join(pair, 'first')
    path2 = os.path.join(pair, 'second')
    output_path = '../outputs/pair_comparison2'
    os.makedirs(output_path, exist_ok=True)
    run_on_pair(path1, path2, output=output_path, cell_size=256, image_size=1024, subsample=4)
    run_on_pair(path1, path2, output=output_path, cell_size=256, image_size=1024, subsample=8)
    run_on_pair(path1, path2, output=output_path, cell_size=128, image_size=1024, subsample=4)
    run_on_pair(path1, path2, output=output_path, cell_size=128, image_size=1024, subsample=8)
    run_on_pair(path1, path2, output=output_path, cell_size=64, image_size=1024, subsample=4)