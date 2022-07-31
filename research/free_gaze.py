import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os
#import cv2
import matplotlib.pyplot as plt
import numpy as np


def run():

    file_path = os.path.join('../parsed_outputs/normal_experiments/ofir_l_2022_06_26_16_54/freegaze.asc')
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


if __name__ == '__main__':
    run()

