import pandas as pd
from argparse import Namespace
import numpy as np
from tqdm import tqdm


class ConversionParams:
    relative_to_initial_position: bool
    relative_to_average: bool
    normalize: bool
    scaleless_normalization: bool
    scaled_normalization: bool
    scaled_normalization_size: float
    add_metadata: bool
    add_speed: bool
    mark_non_samples: bool
    max_samples:int

    def __init__(self,
                 relative_to_initial_position: bool,
                 relative_to_average: bool,
                 normalize: bool,
                 scaleless_normalization: bool,
                 scaled_normalization: bool,
                 scaled_normalization_size: float,
                 add_metadata:bool,
                 add_speed: bool,
                 mark_non_samples: bool,
                 max_samples:int
    ):
        """
        normalize: should the data be normalized
        relative_to_initial_position: should the data be normalized in relation to the initial fixation
        relative_to_average: should the data be normalized in relation to the average location
        scaleless_normalization: always normalize the data to a [-1, 1]
        scaled_normalization: scale the data according to the given size
        add_metadata: add data like std, min, max
        add_speed: add speed data to each sample (i.e. the diff from the last samples)
        """
        self.normalize = normalize
        self.relative_to_initial_position = relative_to_initial_position
        self.relative_to_average = relative_to_average
        self.scaleless_normalization = scaleless_normalization
        self.scaled_normalization = scaled_normalization
        self.scaled_normalization_size = scaled_normalization_size
        self.add_metadata = add_metadata
        self.add_speed = add_speed
        self.mark_non_samples = mark_non_samples
        self.max_samples = max_samples


def convert_fixation(samples:pd.DataFrame, conversion_params: ConversionParams):
    """
    Expects samples with only x and y columns
    """

    keys = ['x', 'y']

    if conversion_params.add_speed:
        samples['dx'] = samples['x'] - samples['x'].shift(-1)
        samples['dy'] = samples['y'] - samples['y'].shift(-1)
        keys = ['x', 'y', 'dx', 'dy']

    normalization_center = [0] * len(samples.keys())
    if conversion_params.relative_to_average:
        normalization_center = samples[keys].mean().values

    if conversion_params.relative_to_initial_position:
        normalization_center = samples.iloc[0].values

    mins, maxs = samples[keys].min().values, samples[keys].max().values

    def scaleless_norm(min_val, max_val, val):
        return (val - min_val) / (max_val - min_val + 1e-6)

    def scaled_norm(center_val, val):
        return (val - center_val) / conversion_params.scaled_normalization_size

    def normalize_sample(sample):
        if not conversion_params.normalize:
            return sample
        if conversion_params.scaleless_normalization:
            return scaleless_norm(mins, maxs, sample)

        if conversion_params.scaled_normalization:
            return scaled_norm(normalization_center, sample)

        return sample

    max_size = conversion_params.max_samples
    data = np.zeros((max_size, len(keys)))
    for idx, (row, sample) in enumerate(samples.iterrows()):
        if idx == max_size:
            break
        normalized = normalize_sample(sample.values)
        data[idx, :] = normalized

    if conversion_params.mark_non_samples:
        marks = np.zeros((max_size, 1))
        marks[idx+1:] = 1
        data = np.concatenate([data, marks], axis=1)
    return data


def load_samples_file(csv_data, params: ConversionParams, ignore_zero_blurs=True):
    if type(csv_data) == str:
        samples = pd.read_csv(csv_data)
    else:
        samples = csv_data
    datas = []
    blurs = []

    for fixation_index in range(1, samples['fixation_index'].max()):
        fixation_samples = samples[samples['fixation_index'] == fixation_index]
        current_blur = fixation_samples.iloc[0]['blur']
        if current_blur == 0.0:
            continue
        blurs.append(current_blur)
        fixation_samples = fixation_samples[['image_x', 'image_y']].rename({'image_x': 'x', 'image_y': 'y'}, axis=1)
        datas.append(convert_fixation(fixation_samples, params))

    return blurs, datas


def load_multiple_files(files: list, params:ConversionParams, ignore_zero_blurs=True):
    blurs = []
    data = []
    all_files_data = [pd.read_csv(f) for f in files]
    for single_file_data in all_files_data:
        new_blurs, new_data = load_samples_file(single_file_data, params, ignore_zero_blurs)
        blurs += new_blurs
        data += new_data

    return blurs, data


if __name__ == '__main__':
    params = ConversionParams(
        relative_to_initial_position=False,
        relative_to_average=True,
        normalize=True,
        scaleless_normalization=False,
        scaled_normalization=True,
        scaled_normalization_size=25,
        add_metadata=True,
        add_speed=True,
        mark_non_samples=True
    )

    paths = [
        '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur/samples.csv',
        '../../../../outputs/preprocessed_outputs/FGBS/pair/first/samples.csv',
        '../../../../outputs/preprocessed_outputs/FGBS/pair/second/samples.csv',

    ]
    blurs, data = load_multiple_files(paths, params=params)

    # print(run_xgboost.run_xgboost(data, blurs))


