import pandas as pd
import numpy as np
from eyelinkparser._eyelinkplusparser import EyeLinkPlusParser

def add_sample_state(parser: EyeLinkPlusParser):
    """
    for each sample in the parser data, add fields that tell if this is part of a fixation, saccade or blink
    """

    samples = parser.sample_df
    zeros = np.zeros(len(samples), dtype=bool)
    samples = samples.assign(
        is_fixation=zeros,
        is_saccade=zeros,
        is_blink=zeros
    )

    for blink in parser.blinks_df.iloc:
        samples.is_blink |= (samples.time >= blink['start time']) & (samples.time <= blink['end time'])

    for saccade in parser.saccades_df.iloc:
        samples.is_saccade |= (samples.time >= saccade['start time']) & (samples.time <= saccade['end time'])

    for fixation in parser.fixations_df.iloc:
        samples.is_fixation |= (samples.time >= fixation['start time']) & (samples.time <= fixation['end time'])

    parser.sample_df = samples


def add_fixation_order(parser: EyeLinkPlusParser):
    fixation_index = np.zeros(len(parser.sample_df), dtype=int)
    was_last_sample_a_fixation = False
    current_fixation_index = 0
    for i, sample in enumerate(parser.sample_df.iloc):
        if sample['is_fixation']:
            if not was_last_sample_a_fixation:
                current_fixation_index += 1
                was_last_sample_a_fixation = True

            fixation_index[i] = current_fixation_index

        else:
            fixation_index[i] = 0
            was_last_sample_a_fixation = False


    parser.sample_df = parser.sample_df.assign(fixation_index=fixation_index)
    parser.metadata.number_of_fixations = fixation_index.max()


def add_image_order(parser: EyeLinkPlusParser):
    zeros = np.zeros(len(parser.sample_df), dtype=int)
    samples = parser.sample_df.assign(
        image_index = zeros
    )
    for index, image in enumerate(parser.images_df.iloc):
        samples.loc[(samples['time'] >= image['start time']) & (samples['time'] <= image['end time']), 'image_index'] = index

    parser.sample_df = samples


def add_image_aligned_position(parser: EyeLinkPlusParser):
    zeros = np.zeros(len(parser.sample_df), dtype=float)
    samples = parser.sample_df.assign(
        image_x = zeros,
        image_y = zeros
    )

    def align_loc(image, x, y):
        image_start = image['location'][0] - (image['size'][0]//2), image['location'][1] - (image['size'][1]//2)
        return x - image_start[0], y - image_start[1]

    for index, image in enumerate(parser.images_df.iloc):
        image_samples = samples.loc[samples['image_index'] == index]['x'], samples.loc[samples['image_index'] == index]['y']
        new_x, new_y = align_loc(image, image_samples[0], image_samples[1])
        samples.loc[samples['image_index'] == index, 'image_x'] = new_x
        samples.loc[samples['image_index'] == index, 'image_y'] = new_y

    parser.sample_df = samples




