import pandas as pd
import numpy as np
from eyelinkparser._eyelinkplusparser import EyeLinkPlusParser

def get_fixation_statistics(parser: EyeLinkPlusParser):
    """
    Expects a parser that has the fixation index data
    """

    fixation_statistics = []
    for fixation_index in range(1, parser.metadata.number_of_fixations):
        fixation_samples = parser.sample_df[parser.sample_df.fixation_index == fixation_index]
        fixation_statistics.append({
            'start': (fixation_samples.iloc[0].x, fixation_samples.iloc[0].y),
            'average': (fixation_samples.x.mean(), fixation_samples.y.mean()),
            'variance': (fixation_samples.x.var(), fixation_samples.y.var())
        })

    return fixation_statistics



