import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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


def get_saccade_statistics(saccades_df:pd.DataFrame, min_speed=0.0, min_distance=0):
    # filter out speed and distance
    s = saccades_df
    s = s.loc[(s['speed'] > min_speed) & (s['distance'] > min_distance)]

    g = s.groupby(by='image_index')

    # number of saccades
    count = pd.DataFrame(g.count()['duration'])
    count.rename(columns={'duration': 'count'}, inplace=True)

    # means
    means = g.mean()
    mean_speed = pd.DataFrame(means['speed'])
    mean_distance = pd.DataFrame(means['distance'])

    # std
    stds = g.std()
    std_speed = pd.DataFrame(stds['speed'])
    std_distance = pd.DataFrame(stds['distance'])

    return count, mean_distance, mean_speed, std_distance, std_speed


def get_blink_count(blinks_df):
    g = blinks_df.groupby(by='image_index')

    # number of saccades
    count = pd.DataFrame(g.count()['duration'])
    count.rename(columns={'duration': 'count'}, inplace=True)
    return count


def add_metric_blur_data(metric:pd.DataFrame, images):
    metric['blur'] = images['blur']
    metric = metric[metric['blur'] != 0]
    return metric



def get_heatmap(xs, ys, sigma=30.0, ksize=91, image_size=1024):
    data = np.zeros((image_size, image_size), dtype=np.float64)
    for x, y in zip(xs, ys):
        data[y, x] += 1

    data = cv2.GaussianBlur(data,ksize=(ksize,ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    return data

def get_images_heatmap(events: pd.DataFrame, sigma=30.0, ksize=91, blur_ranges=[0.1, 5, 10, 15, 20], image_size=1024):
    """
    each event should have a blur field to it, and a x,y field
    """

    bins = []
    for i in range(len(blur_ranges) - 1):
        bins.append(np.zeros((image_size, image_size), dtype=np.float64))

    def get_bin(val):
        for i in range(len(blur_ranges)-1):
            if val < blur_ranges[i+1]:
                return i

    from math import isnan
    for i, row, in events.iterrows():
        blur = row['blur']
        if blur == 0 or isnan(row['image_x']):
            continue
        bin = get_bin(blur)
        x, y= int(row['image_x']), int(row['image_y'])
        if 0 <= x < image_size and 0 <= y < image_size:
            bins[bin][y, x] += 1

    out = [cv2.GaussianBlur(i, ksize=(ksize,ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
           for i in bins]

    return out


if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent.parent.joinpath(Path('outputs')).joinpath(Path('preprocessed_outputs').joinpath('case2'))
    saccades, images, fixations = pd.read_csv(str(path.joinpath('saccades.csv'))), pd.read_csv(str(path.joinpath('images.csv'))), pd.read_csv(str(path.joinpath('fixations.csv')))
    # images = add_image_blur(images)
    metrics = get_saccade_statistics(saccades)

    count, mean_distance, mean_speed, std_distance, std_speed = [add_metric_blur_data(i, images) for i in metrics]

    if False:
        count.plot(x='blur', y='count', kind='scatter')
        plt.title('number of saccades over blur')
        plt.savefig('./outputs/count')

        mean_speed.plot(x='blur', y='speed', kind='scatter')
        plt.title('average speed of saccades over blur')
        plt.savefig('./outputs/avg_speed')

        mean_distance.plot(x='blur', y='distance', kind='scatter')
        plt.title('average distance of saccades over blur')
        plt.savefig('./outputs/avg_distance')

        std_distance.plot(x='blur', y='distance', kind='scatter')
        plt.title('std distance of saccades over blur')
        plt.savefig('./outputs/std_distance')

        std_speed.plot(x='blur', y='speed', kind='scatter')
        plt.title('std speed of saccades over blur')
        plt.savefig('./outputs/std_speed')

    blur_ranges = [0.1, 5, 10, 15, 20]
    fixation_heatmaps = get_images_heatmap(fixations, sigma=30.0, blur_ranges=blur_ranges)
    for idx in range(len(blur_ranges) - 1):
        plt.imshow(fixation_heatmaps[idx], cmap='Reds')
        plt.title(f'fixation heatmap for blurs {blur_ranges[idx]} to {blur_ranges[idx+1]}')
        plt.savefig(f'./outputs/heatmap_{blur_ranges[idx+1]}')

    samples = pd.read_csv(str(path.joinpath('samples.csv')))
    sample_heatmaps = get_images_heatmap(samples, sigma=30.0, blur_ranges=blur_ranges)
    for idx in range(len(blur_ranges) - 1):
        plt.imshow(sample_heatmaps[idx], cmap='Reds')
        plt.title(f'sample heatmap for blurs {blur_ranges[idx]} to {blur_ranges[idx+1]}')
        plt.savefig(f'./outputs/heatmap_sample_{blur_ranges[idx+1]}')


