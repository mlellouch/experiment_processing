import numpy as np

import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
import os
import matplotlib.pyplot as plt


def generate_heat_map(dm):
    size = (2000, 2800)

    data = np.zeros(size)
    for x,y in zip(dm.xtrace_SAMPLES[0], dm.ytrace_SAMPLES[0]):
        if np.isnan(x) or np.isnan(y):
            continue

        if y >= 2000 or x >= 2800:
            continue

        data[int(y),int(x)] += 1


    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()



if __name__ == '__main__':
    absolute_path = os.path.abspath('../data/new_data/circle_2022_03_13_19_01.asc')
    dm = parse_file(
        parser=EyeLinkPlusParser,
        filepath=absolute_path,  # Folder with .asc files
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,  # Interpolate pupil size during blinks
            downsample=1,  # Reduce sampling rate to 100 Hz,
            mode='advanced'  # Use the new 'advanced' algorithm
        )
    )

    generate_heat_map(dm)

    plt.plot(dm.xtrace_SAMPLES[0])
    plt.plot(dm.ytrace_SAMPLES[0])

    plt.legend(["X", "Y"])
    plt.show()