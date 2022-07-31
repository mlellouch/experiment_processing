from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os
import matplotlib as mpl
import seaborn as sns



def run():
    file_path = os.path.join('../../../parsed_outputs/FGB/FGB_2022_07_04_19_19.asc')
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
    a = 1


    #is_saccade_df = parser.sample_df[parser.sample_df['is_saccade'] == True]
    #is_saccade_data = is_saccade_df.groupby('image_index')[['is_saccade']].count()
    #print(is_saccade_data)

    #sns.set_style('darkgrid')
    #sns.barplot(x="image_index", y="num_of_saccades", data='is_saccade_data')
    #mpl.pyplot.show()







if __name__ == '__main__':
    run()

