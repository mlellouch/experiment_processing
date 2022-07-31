from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os

# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run():
    file_path = os.path.join('../parsed_outputs/FGB/FGB_2022_07_04_19_19.asc')
    parser = parse_file(
        parser=EyeLinkPlusParser,
        filepath=file_path,  # Folder with .asc files
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,  # Interpolate pupil size during blinks
            downsample=1,  # Reduce sampling rate to 100 Hz,
            mode='advanced'  # Use the new 'advanced' algorithm
        )
    )

    util.add_sample_state(parser)
    util.add_fixation_order(parser)
    util.add_image_order(parser)
    util.add_image_aligned_position(parser)
    a = 1

    def fixation_overlap(sample_df1, sample_df2, radius):
        # make equal length
        diff = sample_df1.shape[0] - sample_df2.shape[1]
        if diff > 0:
            sample_df1 = sample_df1.loc[0: -diff]
        elif diff < 0:
            sample_df2 = sample_df2.loc[0: diff]
        distances = {}
        # calculates list of distances for each pair of equal index entries
        for index in range(sample_df1.shape[0]):
            if sample_df1.loc[index, 'is_fixation'] and sample_df2.loc[index, 'is_fixation']:
                point1 = np.array((sample_df1.loc[index, 'image_x'], sample_df1.loc[index, 'image_y']))
                point2 = np.array((sample_df2.loc[index, 'image_x'], sample_df2.loc[index, 'image_y']))
                distances[index] = calc_dist(point1, point2)
            else:
                distances[index] = None

    def calc_dist(point1, point2):
        return np.linalg.norm(point1 - point2)


if __name__ == '__main__':
    run()
