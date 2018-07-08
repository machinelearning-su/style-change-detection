import os
import json
from itertools import zip_longest
from time import gmtime, strftime
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

def print_splits(texts, positions):
    text_colors = ['1;31', '1;32', '1;33', '1;34']

    whole_print = []

    for index, text in enumerate(texts):
        positions[index].append(len(text))
        text_marker = 0
        local_print = ''

        for color_index, change in enumerate(positions[index]):
            local_print += '\x1b[%sm%s\x1b[0m' % (text_colors[color_index], text[text_marker:change])
            text_marker = change

        whole_print.append(local_print)

    print('\n\n=============================================\n\n'.join(whole_print))

def get_data(main_dir=None, external_file=None, breach=False):
    x, y, positions, file_names = [], [], [], []
    if main_dir:
        x, y, positions, file_names = get_data_from_dir(main_dir, breach)

    if external_file:
        data = pd.read_feather(external_file)
        external_x = data['text'].values.tolist()
        external_y = [len(x) > 0 for x in data['positions']]
        external_positions = [map(int, x.split(',')) for x in data['positions']]

        x += external_x
        y += external_y
        positions += external_positions

    return x, y, positions, file_names

def get_external_data(file, train_size, val_size):
    data = pd.read_feather(file)
    X = data['text'].values.tolist()
    y = [len(x) > 0 for x in data['positions']]
    
    return train_test_split(X, y, stratify=y, train_size=train_size, test_size=val_size, random_state=2)


def get_data_from_dir(directory, breach=False, size=None):
    x = []
    y = []
    positions = []
    file_names = []
    n = 0

    for entry in os.listdir(directory):
        if n == size:
            break

        root, ext = os.path.splitext(entry)
        if ext == '.txt':
            with open(os.path.join(directory, ''.join([root, ext])), encoding='utf8') as txt_file:
                text = txt_file.read()
                x.append(text)
                file_names.append(root)
                n += 1
            try:
                with open(os.path.join(directory, ''.join([root, '.truth'])), encoding='utf8') as truth_file:
                    truth = json.load(truth_file)
                    if breach:
                        truth_changes = len(truth['borders']) > 0
                        truth_positions = truth['borders']
                    else:
                        truth_changes = truth['changes']
                        truth_positions = truth['positions']
                    y.append(truth_changes)
                    positions.append(truth_positions)
            except IOError:
                pass

                if(size):
                    print_progress_bar(n, size, description = 'Loading artificial data')

    return x, y, positions, file_names


def get_results(train_size, clf_params, cv=None, val=None, gs=None):
    if cv:
        cv = {
            'train_score': {
                "mean": round(np.mean(cv['train_score']), 4),
                "std": round(np.std(cv['train_score']), 2)
            },
            'test_score': {
                "mean": round(np.mean(cv['test_score']), 4),
                "std": round(np.std(cv['test_score']), 2),
                "all": round_np_scores(cv['test_score'], 4)
            },
            'fit_time': humanize_time(max(cv['fit_time'])),
            'score_time': humanize_time(max(cv['score_time']))
        }

    if val:
        val = {
            'accuracy': round(val['accuracy'], 4),
            'time': humanize_time(val['time'])
        }

    results = {
        'cross_validation': cv,
        'validation': val,
        'grid_search': gs,
        'estimator': clf_params,
        'train_size': train_size,
        'timestamp': strftime("%Y-%m-%d %H:%M:%S", gmtime())
    }

    json_results = json.dumps(results, indent=4, sort_keys=True)
    return json_results


def write_results_to_file(results):
    output_separator = '================================================='

    results_path = config_local().get('results_file', None)

    if not results_path:
        print('No file name specified for results!')
        return

    with open(results_path, 'a') as output:
        output.write('%s\n%s\n' % (results, output_separator))

def config_local():
    with open('config.json', 'r') as config_json:
        return json.load(config_json)

def persist_output(output_dir, predictions, file_names, breach=False):
    for prediction, file_name in zip(predictions, file_names):
        if breach:
            prediction = {
                'borders': prediction
            }
        else:
            tag = True if prediction == 1 else False

            prediction = {
                'changes': tag
            }

        json_prediction = json.dumps(prediction, indent=8)

        with open('%s/%s.truth' % (output_dir, file_name), 'w') as output:
            output.write(json_prediction)


def round_np_scores(np_array, p=None):
    return [round(x, p) for x in np_array.tolist()]


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)

    return '%02d:%02d:%02d' % (hours, mins, secs)


def print_progress_bar(iteration, total, description='', decimals=1, bar_length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + '-' * (bar_length - filled_length)

    print('\r |%s| %s%% | %s' %
          (bar, percent, description), end='\r')

    if iteration == total:
        print()


def chunker(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n

    return list(zip_longest(*args, fillvalue=fillvalue))


def get_n_jobs():
    with open('config.json', 'r') as rc:
        n_jobs = json.load(rc).get('n_jobs', 1)

        return n_jobs

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-i", dest="input_dir", help="input_dir", metavar="FILE")
    parser.add_argument("-o", dest="output_dir", help="output_dir", metavar="FILE")
    args = parser.parse_args()

    return args.input_dir, args.output_dir

def update_dict(params, keys, value):
        if len(keys) == 1:
            params[keys[0]] = value
        else:
            update_dict(params[keys[0]], keys[1:], value)
