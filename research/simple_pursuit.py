import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
import os

if __name__ == '__main__':
    absolute_path = os.path.abspath('../data/new_data/sample2_2022_03_03_17_57.asc')
    dm = parse_file(
        parser=EyeLinkPlusParser,
        filepath=absolute_path,  # Folder with .asc files
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,  # Interpolate pupil size during blinks
            downsample=1,  # Reduce sampling rate to 100 Hz,
            mode='advanced'  # Use the new 'advanced' algorithm
        )
    )

    a=1