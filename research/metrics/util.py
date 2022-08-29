import os
import csv
import pandas as pd
import numpy as np
import pathlib
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
    for index, image in parser.images_df.iterrows():
        samples.loc[(samples['time'] >= image['start time']) & (samples['time'] <= image['end time']), 'image_index'] = index

    parser.sample_df = samples


    zeros = np.zeros(len(parser.fixations_df), dtype=int)
    fixations = parser.fixations_df.assign(
        image_index = zeros
    )
    for index, image in parser.images_df.iterrows():
        fixations.loc[(fixations['start time'] >= image['start time']) & (fixations['end time'] <= image['end time']), 'image_index'] = index

    parser.fixations_df = fixations


def add_image_blur(parser: EyeLinkPlusParser):
    blur = []
    for idx, row in parser.images_df.iterrows():
        parts = row.path.split('_')
        if len(parts) <= 1:
            blur.append(0)

        else:
            try:
                blur.append(float(parts[-1][:parts[-1].rfind('.')]))
            except ValueError:
                blur.append(0)

    parser.images_df = parser.images_df.assign(
        blur=np.array(blur)
    )


def add_saccade_description(parser: EyeLinkPlusParser):
    s = parser.saccades_df
    start_points = np.array([s['start x'], s['start y']]).T
    end_points = np.array([s['end x'], s['end y']]).T
    distance = np.linalg.norm(end_points - start_points, axis=1)
    speed = distance / (s['end time'] - s['start time'])
    parser.saccades_df = parser.saccades_df.assign(distance=distance, speed=speed)


def add_image_aligned_position(parser: EyeLinkPlusParser):
    zeros = np.zeros(len(parser.sample_df), dtype=float)
    samples = parser.sample_df.assign(
        image_x=zeros,
        image_y=zeros
    )

    def align_loc(image, x, y):
        image_start = image['location'][0] - (image['size'][0]//2), image['location'][1] - (image['size'][1]//2)
        return x - image_start[0], y - image_start[1]

    for index, image in parser.images_df.iterrows():
        image_samples = samples.loc[samples['image_index'] == index]['x'], samples.loc[samples['image_index'] == index]['y']
        new_x, new_y = align_loc(image, image_samples[0], image_samples[1])
        samples.loc[samples['image_index'] == index, 'image_x'] = new_x
        samples.loc[samples['image_index'] == index, 'image_y'] = new_y

    parser.sample_df = samples

    zeros = np.zeros(len(parser.fixations_df), dtype=float)
    fixations = parser.fixations_df.assign(
        image_x=zeros,
        image_y=zeros
    )
    for index, image in parser.images_df.iterrows():
        image_fixations = fixations.loc[fixations['image_index'] == index]['x'], fixations.loc[fixations['image_index'] == index]['y']
        new_x, new_y = align_loc(image, image_fixations[0], image_fixations[1])
        fixations.loc[fixations['image_index'] == index, 'image_x'] = new_x
        fixations.loc[fixations['image_index'] == index, 'image_y'] = new_y

    parser.fixations_df = fixations


def add_image_to_events_data(parser: EyeLinkPlusParser, add_image_blur=False):

    def add_image_to_event_data(images_df: pd.DataFrame, events_df: pd.DataFrame):
        image_index = 0
        new_image_column = []
        new_blur_column = []

        for i, event in events_df.iterrows():
            found = False
            while True:
                if image_index >= len(images_df):
                    # we finished going through all images
                    break

                if images_df.iloc[image_index]['start time'] <= event['start time'] < images_df.iloc[image_index]['end time']:
                    found = True
                    break

                if event['start time'] <= images_df.iloc[image_index]['start time']: # if not in current image, but is before next one:
                    # then this event is not in any image
                    break

                # this event happens after current image. hence we advance the image counter
                image_index += 1

            new_image_column.append(image_index) if found else new_image_column.append(0)
            if add_image_blur:
                new_blur_column.append(parser.images_df.loc[image_index]['blur']) if found else new_blur_column.append(0)


        return new_image_column, new_blur_column

    blinks, blur = add_image_to_event_data(parser.images_df, parser.blinks_df)
    blinks_df = parser.blinks_df.assign(image_index=blinks, blur=blur)
    parser.blinks_df = blinks_df

    saccades, blur = add_image_to_event_data(parser.images_df, parser.saccades_df)
    saccades_df = parser.saccades_df.assign(image_index=saccades, blur=blur)
    parser.saccades_df = saccades_df

    fixations, blur = add_image_to_event_data(parser.images_df, parser.fixations_df)
    fixations_df = parser.fixations_df.assign(image_index=fixations, blur=blur)
    parser.fixations_df = fixations_df

    # add blur to samples
    zeros = np.zeros(len(parser.sample_df))
    samples = parser.sample_df.assign(blur=zeros)
    for idx, image in parser.images_df.iterrows():
        samples.loc[samples['image_index'] == idx, 'blur'] = parser.images_df.loc[idx]['blur']

    parser.sample_df = samples


def subject_to_acuity(subject):
    current_path = pathlib.Path(__file__).parent.parent.parent.joinpath('metadata.csv')
    with open(str(current_path), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] in subject:
                return row['logMAR']

    return 1


def get_all_experiments(experiment_name):
    """
    a generator of all experiements of a single case
    """

    experiment_name += '.asc'
    current_path = pathlib.Path(__file__).parent.resolve()
    experiments_path = current_path.parent.parent.joinpath(pathlib.Path('parsed_outputs')).joinpath(pathlib.Path('normal_experiments'))
    for subject in experiments_path.iterdir():
        acuity = subject_to_acuity(str(subject))
        yield acuity, str(subject.joinpath(experiment_name))




