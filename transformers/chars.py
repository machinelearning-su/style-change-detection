import re
from utils import print_progress_bar, chunker

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

def ascii_discrepancies(data, window, local_diff, feature_names=[]):
    vectors = []

    data_length = len(data)

    isascii = lambda s: len(s) == len(s.encode())

    for i, entry in enumerate(data):
        entry = entry.lower()

        window_chars = round(len(entry) * window)

        local = []

        for chunk in chunker(entry, window_chars):
            non_ascii_count = 0

            for char in chunk:
                if not char: continue

                if not isascii(char):
                    non_ascii_count += 1

            local.append([non_ascii_count / len(chunk)])

        if(local_diff):
            local_len = len(local)

            for local_index in range(local_len):
                if(local_index == local_len - 1): break

                local[local_index] = [abs(a - b) for a, b in zip(local[local_index], local[local_index + 1])]

        min_v = np.amin(local, axis=0).tolist()
        max_v = np.amax(local, axis=0).tolist()
        diff = np.subtract(max_v, min_v).tolist()

        vectors.append(diff)

        print_progress_bar(i + 1, data_length, description = 'ascii_chars_discrepancies')

    
    feature_names.extend(['ascii_chars_discrepancies'])

    return vectors

def text_length(data, feature_names=[]):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        vectors.append([float(len(entry))])

        print_progress_bar(i + 1, data_length, description = 'text_length')
    
    
    feature_names.extend(['text_length'])

    return vectors

def num_paragraphs(data, feature_names=[]):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        vectors.append([float(entry.count('\n') + 1)])

        print_progress_bar(i + 1, data_length, description = 'num_paragraphs')
    
    
    feature_names.extend(['num_paragraphs'])

    return vectors
