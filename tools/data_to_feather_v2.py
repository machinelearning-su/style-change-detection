import pandas as pd
import json, os.path
from utils import print_progress_bar

DIR = 'data/training_artificial/'
FEATHER_FILE = 'external_feather'
written_count = 0
df = pd.DataFrame(columns=['text', 'positions'])
size = len(os.listdir(DIR)) / 2

for entry in os.listdir(DIR):
    root, ext = os.path.splitext(entry)
    if ext != '.txt': continue

    with open(os.path.join(DIR, ''.join([root, ext]))) as txt_file:
        text = txt_file.read()
    try:
        with open(os.path.join(DIR, ''.join([root, '.truth']))) as truth_file:
            truth = json.load(truth_file)
            df.loc[written_count] = [text, ','.join(map(str, truth['positions']))]
            written_count += 1
    except IOError: pass
    print_progress_bar(written_count, size, description = 'Feathering artificial data')

df.to_feather(FEATHER_FILE)
