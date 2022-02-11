import unittest
import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor
from eyelinkparser import EyeLinkPlusParser
import os


class NewDataParsing(unittest.TestCase):

    def test_sanity(self):
        absolute_path = os.path.abspath('../data/new_data')
        dm = parse(
            parser=EyeLinkPlusParser,
            folder=absolute_path,  # Folder with .asc files
            traceprocessor=defaulttraceprocessor(
                blinkreconstruct=True,  # Interpolate pupil size during blinks
                downsample=10,  # Reduce sampling rate to 100 Hz,
                mode='advanced'  # Use the new 'advanced' algorithm
            )
        )

    def test_trials(self):
        absolute_path = os.path.abspath('../data/new_data')
        dm = parse(
            parser=EyeLinkPlusParser,
            folder=absolute_path,  # Folder with .asc files
            traceprocessor=defaulttraceprocessor(
                blinkreconstruct=True,  # Interpolate pupil size during blinks
                downsample=10,  # Reduce sampling rate to 100 Hz,
                mode='advanced'  # Use the new 'advanced' algorithm
            )
        )

        assert hasattr(dm, "trialid")
        assert len(dm.trialid) == 4





if __name__ == '__main__':
    unittest.main()
