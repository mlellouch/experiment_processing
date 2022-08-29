from eyelinkparser._eyelinkparser import EyeLinkParser
from eyelinkparser._events import event, sample, fixation, saccade, blink, Saccade
import warnings
import os
import pandas as pd


class ParseMetadata:

    def __init__(self):
        self.number_of_fixations = 0

class EyeLinkPlusParser(EyeLinkParser):

    def __init__(self, *args, **kwargs):
        self.saccades = []
        self.samples = []
        self.blinks = []
        self.fixations = []
        self.images= [{'start time': 0, 'path': 'no image', 'location': (0,0), 'size': (0,0)}]
        self.end_phase_called = False
        super(EyeLinkPlusParser, self).__init__(*args, **kwargs)
        self.last_trial_result = -1
        self.metadata = ParseMetadata()

    def start_phase(self, l):
        super(EyeLinkPlusParser, self).start_phase(l)
        self.saccade_x_list = []
        self.saccade_y_list = []


    def is_start_trial(self, l):
        # catch messages like "MSG	1559086 TRIALID 4"

        if self.match(l, u'MSG', int, u'TRIALID', int):
            self.trialid = int(l[3])
            self.current_phase = None
            return True

        return False

    def is_end_trial(self, l):
        # catch message like "MSG	1528867 TRIAL_RESULT 0"
        if self.match(l, u'MSG', int, u'TRIAL_RESULT', int):
            self.last_trial_result = int(l[3])
            return True

        return False

    def parse_saccade(self, s: Saccade):
        self.saccades.append(
            {
                'duration': s.duration,
                'start time': s.st,
                'end time': s.et,
                'start x': s.sx,
                'end x': s.ex,
                'start y': s.sy,
                'end y': s.ey,
                'size': s.size,
            }
        )

    def parse_sample(self, s):
        self.samples.append({
            'time': s.t,
            'pupil size': s.pupil_size,
            'x': s.x,
            'y': s.y
        })

    def parse_fixation(self, f):
        self.fixations.append({
            'x': f.x,
            'y': f.y,
            'start time': f.st,
            'end time': f.et
        })

    def parse_blink(self, b):
        self.blinks.append({
            'start time': b.st,
            'end time': b.et
        })

    def parse_message(self, msg):
        time = int(msg[1])
        if len(msg) <= 3:
            return

        if msg[3] == 'IMGLOAD':
            if len(self.images) != 0:
                self.images[-1]['end time'] = time
            self.images.append({
                'start time': time,
                'path': os.path.basename(msg[5]),
                'location': (msg[6], msg[7]),
                'size': (msg[8], msg[9])
            })

        if msg[3] == 'CLEAR':
            if len(self.images) != 0:
                self.images[-1]['end time'] = time


    def parse_phase(self, l):
        # for our case, we actually just parse a trial
        if l[0] == 'START':
            self.start_phase(l)
            return

        if l[0] == 'END':
            self.end_phase(l)
            return

        if l[0] == 'PRESCALER' or l[0] == 'VPRESCALER':
            self.prescaler = int(l[1])
            return

        if l[0] == 'MSG':
            self.parse_message(l)

        s = sample(l)
        if s is not None:
            self.parse_sample(s)
            return
        f = fixation(l)
        if f is not None:
            self.parse_fixation(f)

        b = blink(l)
        if b is not None:
            self.parse_blink(b)

        s = saccade(l)
        if s is not None:
            self.parse_saccade(s)


    def end_phase(self, l):
        if self.end_phase_called:
            return
        self.end_phase_called = True
        self.saccades_df = pd.DataFrame.from_dict(self.saccades)
        self.saccades_df.sort_values(by='start time', inplace=True)
        del self.saccades

        self.blinks_df = pd.DataFrame.from_dict(self.blinks)
        self.blinks_df.sort_values(by='start time', inplace=True)
        del self.blinks

        self.fixations_df = pd.DataFrame.from_dict(self.fixations)
        self.fixations_df.sort_values(by='start time', inplace=True)
        del self.fixations

        self.sample_df = pd.DataFrame.from_dict(self.samples)
        self.sample_df.sort_values(by='time', inplace=True)
        del self.samples

        self.images_df = pd.DataFrame.from_dict(self.images)
        self.images_df.sort_values(by='start time', inplace=True)
        del self.images

