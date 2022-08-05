import pandas as pd
import numpy as np
from pathlib import Path


def add_image_blur(images_df:pd.DataFrame):
    blur = []
    for idx, row in images_df.iterrows():
        parts = row.path.split('_')
        if len(parts) <= 1:
            blur.append(0)

        else:
            blur.append(float(parts[1][:parts[1].rfind('.')]))

    return images_df.assign(
        blur=np.array(blur)
    )


def get_saccade_statistics(images_df: pd.DataFrame, saccades_df:pd.DataFrame):
    # number of saccades
    g = saccades.groupby(by='image_index')
    count = pd.DataFrame(g.count()['duration'])
    count.rename(columns = {'duration': 'count'}, inplace=True)





if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent.parent.joinpath(Path('outputs')).joinpath(Path('preprocessed_outputs').joinpath('case2'))
    saccades, images = pd.read_csv(str(path.joinpath('saccades.csv'))), pd.read_csv(str(path.joinpath('images.csv')))
    images = add_image_blur(images)
    get_saccade_statistics(images, saccades)