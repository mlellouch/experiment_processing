from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os



def run():
    file_path = os.path.join('../../../outputs/parsed_outputs/FGBS/FGBS_2022_08_21_14_04.asc')
    parser = parse_file(
        parser=EyeLinkPlusParser,
        filepath=file_path,  # Folder with .asc files
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,  # Interpolate pupil size during blinks
            downsample=1,  # Reduce sampling rate to 100 Hz,
            mode='advanced'  # Use the new 'advanced' algorithm
        )
    )

    util.add_image_blur(parser)
    util.add_saccade_description(parser)
    util.add_sample_state(parser)
    util.add_fixation_order(parser)
    util.add_image_order(parser)
    util.add_image_aligned_position(parser)
    util.add_image_to_events_data(parser, add_image_blur=True)
    return parser

if __name__ == '__main__':
    parser = run()
    output_path = '../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur'
    parser.fixations_df.to_csv(os.path.join(output_path, 'fixations.csv'))
    parser.sample_df.to_csv(os.path.join(output_path, 'samples.csv'))
    parser.blinks_df.to_csv(os.path.join(output_path, 'blinks.csv'))
    parser.saccades_df.to_csv(os.path.join(output_path, 'saccades.csv'))
    parser.images_df.to_csv(os.path.join(output_path, 'images.csv'))



