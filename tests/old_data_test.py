import unittest
import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor
import os


class OldDataParsing(unittest.TestCase):

    def test_trials(self):
        """
        Test that the library parses out the trials
        """
        absolute_path = os.path.abspath('../data/old_data')
        dm = parse(
            folder=absolute_path,  # Folder with .asc files
            traceprocessor=defaulttraceprocessor(
                blinkreconstruct=True,  # Interpolate pupil size during blinks
                downsample=10,  # Reduce sampling rate to 100 Hz,
                mode='advanced'  # Use the new 'advanced' algorithm
            )
        )

        assert hasattr(dm, "trialid")
        assert len(dm.trialid) > 0





if __name__ == '__main__':
    unittest.main()
