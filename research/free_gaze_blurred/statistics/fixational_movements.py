import pandas as pd
import os
import matplotlib.pyplot as plt

def samples_std(samples: pd.DataFrame):
    x_std = samples['image_x'].std()
    y_std = samples['image_y'].std()
    return (x_std + y_std) / 2


def get_fixations_std(samples):
    max_fixation = samples['fixation_index'].max()
    stds = []
    blurs = []
    for fixation in range(1, max_fixation):
        fixation_samples = samples[samples['fixation_index'] == fixation]
        blur = fixation_samples.iloc[0]['blur']
        if blur == 0.0:
            continue
        stds.append(samples_std(fixation_samples))
        blurs.append(blur)

    stds = pd.DataFrame(stds, columns=['std'])
    blurs = pd.DataFrame(blurs, columns=['blur'])
    return stds.join(blurs)

def fixational_movements_analysis(path, output):
    samples = pd.read_csv(os.path.join(path, 'samples.csv'))
    stds = get_fixations_std(samples)

    stds.boxplot(column='std', by='blur')
    plt.title('Fixational std over blur')
    plt.savefig(os.path.join(output, 'fixational_std'))
    plt.clf()


if __name__ == '__main__':
    path = '../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur'
    output = './outputs/scenes_aligned'
    os.makedirs(output, exist_ok=True)
    fixational_movements_analysis(path, output)
