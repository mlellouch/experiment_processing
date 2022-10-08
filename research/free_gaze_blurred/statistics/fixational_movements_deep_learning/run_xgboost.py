import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from build_dataset import load_multiple_files, ConversionParams
from itertools import product
import csv
from tqdm import tqdm
import math


def blurs_to_labels(blurs: list):
    sorted_blurs = list(set(blurs.copy()))
    sorted_blurs.sort()
    return [sorted_blurs.index(b) for b in blurs]


def run_xgboost(data, blurs):
    X = data
    Y = blurs_to_labels(blurs)
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = X_train.reshape((-1, X_train.shape[1] * X_train.shape[2]))
    X_test = X_test.reshape((-1, X_test.shape[1] * X_test.shape[2]))

    # fit model no training data
    model = XGBClassifier()
    cls = model.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.Figure()
    sns.heatmap(conf_matrix)
    plt.show()
    return accuracy_score(y_pred, y_test)


def find_best_conversion():
    paths = [
        '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur/samples.csv',
        '../../../../outputs/preprocessed_outputs/FGBS/pair/first/samples.csv',
    ]

    all_possible_params = [
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [10.0, 25.0, 35.0],
        [False],
        [False, True],
        [False, True],
        [25, 50, 100]
    ]

    number_of_combinations = math.prod([len(p) for p in all_possible_params])
    with open('options.csv', 'w', newline='') as f:
        fieldnames = [
            'relative_to_initial_position',
            'relative_to_average',
            'normalize',
            'scaleless_normalization',
            'scaled_normalization',
            'scaled_normalization_size',
            'add_metadata',
            'add_speed',
            'mark_non_samples',
            'max_samples',
            'accuracy'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for param_combination in tqdm(product(*all_possible_params), total=number_of_combinations):
            params = ConversionParams(*param_combination)
            blurs, data = load_multiple_files(paths, params=params, ignore_zero_blurs=True)
            acc = run_xgboost(data, blurs)
            row = {key: val for key, val in zip(fieldnames, list(param_combination) + [acc,])}
            writer.writerow(row)


if __name__ == '__main__':
    paths = [
        '../../../../outputs/preprocessed_outputs/FGBS/case3_aligned_blur/samples.csv',
        '../../../../outputs/preprocessed_outputs/FGBS/pair/first/samples.csv',
    ]

    params = ConversionParams(False, False, False, False, False, 10, False, False, True, 25)
    blurs, data = load_multiple_files(paths, params=params, ignore_zero_blurs=True)
    acc = run_xgboost(data, blurs)

    # find_best_conversion()