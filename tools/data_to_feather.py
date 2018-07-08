import pandas as pd
import json
import os.path

TEXT_EXTENSION = ".txt"
TRUTH_EXTENSION = ".truth"
PROBLEM_PREFIX = "problem-"
DIR = "../training_external/"
TEXT_AS_INDEX = "text"
POSITION_AS_INDEX = "positions"

ITERATIONS = 300000 # it depends on the total number of files
FEATHER_FILE = "external_data_feather"

skeleton = {TEXT_AS_INDEX: [], POSITION_AS_INDEX: []}

data = pd.DataFrame(skeleton)

for file_id in range(1, ITERATIONS):

    TRAIN_FILE = DIR + PROBLEM_PREFIX + str(file_id) + TEXT_EXTENSION

    if os.path.isfile(TRAIN_FILE):
        TRUTH_FILE = DIR + PROBLEM_PREFIX + str(file_id) + TRUTH_EXTENSION

        temporary_file = json.load(open(TRUTH_FILE))
        positions = ','.join(str(position) for (position) in temporary_file[POSITION_AS_INDEX])

        with open(TRAIN_FILE, 'r') as content_file:
            content = content_file.read()

        bufferDataFrame = pd.DataFrame({TEXT_AS_INDEX: [content], POSITION_AS_INDEX: [positions]})
        data = data.append(bufferDataFrame)

    if file_id % 10000 == 0:
        print(file_id)

data.reset_index().to_feather(FEATHER_FILE)
print("DONE")
