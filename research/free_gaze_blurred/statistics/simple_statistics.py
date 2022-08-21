import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


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

def get_images_heatmap(events: pd.DataFrame, sigma=30.0, ksize=91, image_size=1024):
    """
    each event should have a blur field to it, and a x,y field
    """

    bins = {}
    from math import isnan
    for i, row, in events.iterrows():
        blur = row['blur']
        if blur == 0 or isnan(row['image_x']):
            continue

        if blur not in bins.keys():
            bins[blur] = np.zeros((image_size, image_size), dtype=np.float64)

        x, y = int(row['image_x']), int(row['image_y'])
        if 0 <= x < image_size and 0 <= y < image_size:
            bins[blur][y, x] += 1

    out = {i: cv2.GaussianBlur(bins[i], ksize=(ksize,ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
           for i in bins}

    return out

def add_sample_distance(samples:pd.DataFrame):
    s_x, s_y = samples['x'], samples['y']
    e_x, e_y = s_x.to_list(), s_y.to_list()
    e_x = e_x[1: ]
    e_y = e_y[1: ]
    e_x.append(e_x[-1])
    e_y.append(e_y[-1])

    start = np.stack([s_x, s_y]).T
    end = np.stack([np.array(e_x), np.array(e_y)]).T
    distance = np.linalg.norm(end - start, axis=1)
    return samples.assign(distance=distance)


def get_distance_by_blur(samples, min_distance=0.0, max_distance=100.0):
    bins = {}
    from math import isnan
    for i, row, in samples.iterrows():
        blur = row['blur']
        distance = row['distance']
        if blur == 0 or isnan(row['image_x']) or (distance < min_distance or distance > max_distance):
            continue

        if blur not in bins.keys():
            bins[blur] = []

        bins[blur].append(distance)

    return bins


def run_all(path, output_path):
    saccades, images, fixations = pd.read_csv(str(path.joinpath('saccades.csv'))), pd.read_csv(
        str(path.joinpath('images.csv'))), pd.read_csv(str(path.joinpath('fixations.csv')))
    samples = pd.read_csv(str(path.joinpath('samples.csv')))
    # images = add_image_blur(images)
    metrics = get_saccade_statistics(saccades)

    count, mean_distance, mean_speed, std_distance, std_speed = [add_metric_blur_data(i, images) for i in metrics]

    count.boxplot(column='count', by='blur')
    plt.title('number of saccades over blur')
    plt.savefig(os.path.join(output_path, 'count'))
    plt.clf()

    mean_speed.boxplot(column='speed', by='blur')
    plt.title('average speed of saccades over blur')
    plt.savefig(os.path.join(output_path, 'avg_speed'))
    plt.clf()

    mean_distance.boxplot(column='distance', by='blur')
    plt.title('average distance of saccades over blur')
    plt.savefig(os.path.join(output_path, 'avg_distance'))
    plt.clf()

    std_distance.boxplot(column='distance', by='blur')
    plt.title('std distance of saccades over blur')
    plt.savefig(os.path.join(output_path, 'std_distance'))
    plt.clf()

    std_distance.boxplot(column='distance', by='blur')
    plt.title('std speed of saccades over blur')
    plt.savefig(os.path.join(output_path, 'std_speed'))
    plt.clf()

    fixation_heatmaps = get_images_heatmap(fixations, sigma=30.0)
    for idx in fixation_heatmaps:
        plt.imshow(fixation_heatmaps[idx], cmap='Reds')
        plt.title(f'fixation heatmap for blur {idx}')
        plt.savefig(os.path.join(output_path, f'heatmap_{idx}.png'))
        plt.clf()

    sample_heatmaps = get_images_heatmap(samples, sigma=30.0)
    for idx in sample_heatmaps:
        plt.imshow(sample_heatmaps[idx], cmap='Reds')
        plt.title(f'sample heatmap for blur {idx}')
        plt.savefig(os.path.join(output_path, f'heatmap_sample_{idx}.png'))
        plt.clf()

    samples = add_sample_distance(samples)
    distances = get_distance_by_blur(samples, min_distance=0.0, max_distance=100)
    num_bins = 100

    for idx in distances:
        n, bins, patches = plt.hist(distances[idx], num_bins, facecolor='blue', alpha=0.5)
        plt.ylim((0, 600))
        plt.title(f'Distance histogram for blur {idx}')
        plt.savefig(os.path.join(output_path, f'distance_histograms_{idx}.png'))
        plt.clf()


if __name__ == '__main__':
    path = Path('../../../outputs/preprocessed_outputs/FGB/case6_long_aligned')
    run_all(path, './outputs/long_face_aligned')
