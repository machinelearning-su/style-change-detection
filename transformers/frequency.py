import re
from utils import print_progress_bar

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

FUNCTION_WORDS_FILE = 'data/external/function_words/output_words.txt'

def frequent_words_diff(data, window, use_func_words, use_stop_words, local_diff, feature_names=[]):
    frequent_words = []

    if(use_func_words):
        with open(FUNCTION_WORDS_FILE) as f:
            frequent_words = [word.lower() for word in f.read().splitlines()]
    else:
        use_stop_words = True

    if(use_stop_words):
        frequent_words = list(set(stopwords.words('english') + frequent_words))

    vectors = []

    data_length = len(data)
    for i, entry in enumerate(data):
        entry = re.sub("[^a-zA-Z]+", " ", entry).lower()

        words = word_tokenize(entry)

        window_words = round(len(words) * window)

        local = []

        index = 0
        while(index <= len(words) - window_words):
            tmp = []
            for freq_word in frequent_words:
                tmp.append(float(words[index:index + window_words].count(freq_word)))
            index += window_words
            local.append(tmp)

        if(local_diff):
            local_len = len(local)
            tmp = []
            for local_index in range(local_len):
                if(local_index == local_len - 1): break

                tmp.append([float(abs(a - b)) for a, b in zip(local[local_index], local[local_index + 1])])
            local = tmp

        min_v = np.amin(local, axis=0).tolist()
        max_v = np.amax(local, axis=0).tolist()
        diff = np.subtract(max_v, min_v).tolist()

        vectors.append(diff)

        print_progress_bar(i + 1, data_length, description = 'frequent_words_diff')

    feature_names.extend(frequent_words)

    return vectors
