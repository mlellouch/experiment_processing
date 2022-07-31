import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util
import os

def load_file(filepath):
    absolute_path = os.path.abspath(filepath)
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
    util.add_fixation_order(parser)
    return parser


if __name__ == '__main__':
    for acuity, experiment in util.get_all_experiments('shapes'):
        current_data = load_file(experiment)

