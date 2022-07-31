from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os

# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns


def run():
    file_path = os.path.join('../parsed_outputs/FGB/FGB_2022_07_04_19_19.asc')
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

    blinks_df = parser.blinks_df
    images_df = parser.images_df

    for index, image_row in images_df.iterrows():
        path = image_row['path']
        if len(path.split('_')) > 1:
            images_df.loc[index, 'blur_level'] = path.split('_')[1][0: -4]
        else:
            images_df.loc[index, 'blur_level'] = 0
    images_df = images_df.astype({"blur_level": float}, errors='raise')

    for index, blink_row in blinks_df.iterrows():
        for index1, image_row in images_df.iterrows():
            blink_interval = pd.Interval(blink_row['start time'], blink_row['end time'])
            image_interval = pd.Interval(image_row['start time'], image_row['end time'])
            if blink_interval.overlaps(image_interval):
                blinks_df.loc[index, 'image_index'] = index1
                blinks_df.loc[index, 'image_blur'] = images_df.loc[index1, 'blur_level']
    blinks_df = blinks_df.astype({"image_index": int}, errors='raise')



    blinks_per_image = blinks_df.groupby(['image_index'], as_index=False).count()
    for index, row in blinks_per_image:
        blinks_per_image.loc[index, 'image_blur'] = images_df.loc[row['image_index'], 'blur_level']
    print(blinks_per_image)

    # blinks_df.groupby.plot.scatter(x='x_column_name', y='y_columnn_name')
    print(blinks_df)
    print(images_df)



    '''
    images_df = parser.images_df
    images_df[['path']]
    
    sns.set_style('darkgrid')
    sns.barplot(x=is_blink_count['image_index'], y=is_blink_count['is_blink'])
    mpl.pyplot.show()
    '''

    # print(is_blink_count)


if __name__ == '__main__':
    run()
