from eyelinkparser._eyelinkparser import EyeLinkParser

class EyeLinkPlusParser(EyeLinkParser):

    def __init__(self, *args, **kwargs):
        super(EyeLinkPlusParser, self).__init__(*args, **kwargs)
        self.last_trial_result = -1


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
