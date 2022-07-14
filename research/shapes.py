import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util
import os

if __name__ == '__main__':
    absolute_path = os.path.abspath('../parsed_outputs/normal_experiments/313547341_left_2022_06_19_16_47/shapes.asc')
    parser = parse_file(
        parser=EyeLinkPlusParser,
        filepath=absolute_path,  # Folder with .asc files
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,  # Interpolate pupil size during blinks
            downsample=1,  # Reduce sampling rate to 100 Hz,
            mode='advanced'  # Use the new 'advanced' algorithm
        )
    )

    dm = parser.dm
    util.add_sample_state(parser)
    a = 1
