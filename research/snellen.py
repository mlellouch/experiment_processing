import eyelinkparser as ep
from eyelinkparser import parse, defaulttraceprocessor, parse_file
from eyelinkparser import EyeLinkPlusParser
from research.metrics import util, fixation_analysis
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_y_location_to_var(parser):
    fixation_stats = fixation_analysis.get_fixation_statistics(parser)
    background_image = '../../custom_experiments/snellen/snellen_9_feet.png'
    img = cv2.imread(background_image)
    image_center = 1920, 1080

    def screen_location_to_image_location(loc):
        image_start = image_center[0] - img.shape[0] / 2, image_center[1] - img.shape[1] / 2
        return int(loc[0] - image_start[0]), int(loc[1] - image_start[1])

    fixation_stats.sort(key=lambda val: val['average'][1])
    graph_x = []
    graph_y = []
    for i, fixation in enumerate(fixation_stats):
        average_var = (fixation['variance'][0] + fixation['variance'][1]) / 2
        if average_var > 500:
            continue

        y_loc = screen_location_to_image_location(fixation['average'])[1]
        graph_x.append(y_loc)
        graph_y.append(average_var)

    graph_x, graph_y = np.array(graph_x), np.array(graph_y)
    plt.plot(graph_x, graph_y)

def run():
    all_subjects = os.listdir('../parsed_outputs')
    for subject in all_subjects:
        snellen_path = os.path.join('../parsed_outputs', subject, 'snellen.asc')
        parser = parse_file(
            parser=EyeLinkPlusParser,
            filepath=snellen_path,  # Folder with .asc files
            traceprocessor=defaulttraceprocessor(
                blinkreconstruct=True,  # Interpolate pupil size during blinks
                downsample=1,  # Reduce sampling rate to 100 Hz,
                mode='advanced'  # Use the new 'advanced' algorithm
            )
        )

        util.add_sample_state(parser)
        util.add_fixation_order(parser)
        plot_y_location_to_var(parser)
        plt.savefig(os.path.join('./research_outputs/snellen', subject))
        plt.close()

if __name__ == '__main__':
    run()



    # for stat in fixation_stats:
    #     center = screen_location_to_image_location(stat['average'])
    #     axes = int(stat['variance'][0] // 3), int(stat['variance'][1] // 3)
    #     img = cv2.ellipse(img, center=center, axes=axes, angle=0.0, startAngle=0., endAngle=360.0, color=(255, 0, 0))
    # plt.imshow(img)
    # plt.show()
