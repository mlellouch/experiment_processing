import research.transformer.generate_text as generate_text
import pandas as pd
import numpy as np
import os
from argparse import Namespace
from tqdm import tqdm
from itertools import product
import editdistance
import seaborn as sns
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    source = '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur'
    dest = '../outputs/scenes_aligned/'
    run(source, dest, only_for_same_images=False)