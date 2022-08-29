from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os
import itertools
from string import ascii_lowercase
import configargparse

def parse_args():
    parser = configargparse.get_arg_parser()
    parser.add_argument('--new_line', action='store_true', help='should each sample be printed in a new line')
    parser.add_argument('--fixations_only', action='store_true', help='should only the fixations be written')
    parser.add_argument('--skip_saccade_samples', action='store_true', help='skip all samples that are part of a saccade')
    parser.add_argument('--special_saccade_encode', action='store_true', help='should saccades be prepended with a special prefix')
    parser.add_argument('--encode_time', action='store_true', help='should time also be encoded')
    parser.add_argument('--cell_based_location', action='store_true', help='should location be encoded as a cell')
    parser.add_argument('--cell_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=1024)
    return parser.parse_args()


def load_asc_file(file_path):
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
    return parser


def build_image_cells(image_size, cell_size):
    strings = itertools.product(*[ascii_lowercase]*3)
    cells = {}
    for x in range(0, image_size-1, cell_size):
        for y in range(0, image_size-1, cell_size):
            next_string = ''.join(next(strings))
            cells[(x, x+cell_size, y, y+cell_size)] = next_string

    return cells

def location_to_cell(location, cells:dict):
    for key in cells.keys():
        if key[0] <= location[0] < key[1] and key[2] <= location[1] < key[3]:
            return cells[key]

    return None


class SampleEncoder:
    """
    Converts samples to text
    """

    def __init__(self, args):
        self.args = args
        self.cells = build_image_cells(args.image_size, args.cell_size)
        self.last_fixation = -1
        self.last_time = 0

    def _encode_sample(self, sample):
        sample_sentence = location_to_cell((sample.image_x, sample.image_y), self.cells) if self.args.cell_based_location else f'{int(sample.image_x)} {int(sample.image_y)}'
        if sample_sentence is None:
            sample_sentence = 'O' # for outside
        if self.args.encode_time:
            sample_sentence += f' {int(sample.time - self.last_time)}'

        return sample_sentence

    def encode(self, sample):
        is_new_fixation = False
        if self.last_fixation != sample.fixation_index:
            is_new_fixation = True

        if self.args.fixations_only and not is_new_fixation:
            return ''

        text_sample = self._encode_sample(sample)
        if self.args.special_saccade_encode and is_new_fixation:
            text_sample = 'S ' + text_sample

        if args.new_line:
            text_sample += '\n'

        self.last_time = sample.time
        self.last_fixation = sample.fixation_index
        return text_sample




def image_to_text(args, dst_path:str, parser, image_index):
    samples = parser.sample_df[parser.sample_df['image_index'] == image_index]
    encoder = SampleEncoder(args)
    with open(dst_path, 'w') as f:
        for sample in samples.iloc:
            f.write(encoder.encode(sample))



def generate_text_files(args, source_file:str, output_path:str):
    parser = load_asc_file(source_file)
    output_prefix = os.path.basename(source_file)
    output_prefix = output_prefix[:output_path.rfind('.')]
    output_prefix = os.path.join(output_path, output_prefix)

    for index, image in enumerate(parser.images_df.iloc):
        if index == 0:
            continue

        # extract samples
        image_samples = parser.sample_df.loc[parser.sample_df['image_index'] == index]
        output_file = output_prefix + image['path'].replace('jpg', 'txt')
        image_to_text(args, dst_path=output_file, parser=parser, image_index=index)





if __name__ == '__main__':
    args = parse_args()
    generate_text_files(args, source_file='../../parsed_outputs/FGB/FGB_2022_07_04_19_19.asc', output_path='../research_outputs/transformer')



