import re
from utils import print_progress_bar

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

def phrase_frequency(data, word_gram_sizes, stop_words, use_mean, feature_names=[]):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        words = word_tokenize(entry)

        if(stop_words):
            words = remove_stop_words(words)

        local = []
        for word_gram_size in word_gram_sizes:
            local.append(get_ordered_words_occurances(words, entry, word_gram_size, use_mean))

        vectors.append(local)

        print_progress_bar(i + 1, data_length, description = 'phrase_frequency')

    feature_names.extend([str(size) + 'gram' for size in word_gram_sizes])

    return vectors

def get_ordered_words_occurances(words, text, window, use_mean):
    count_words = len(words)

    local = []

    for index in range(count_words):
        if(index == count_words - window + 1): break

        current_window = ' '.join(words[index:index + window])
        local.append(text.count(current_window))

    if(use_mean):
        return sum(local) / float(len(local))

    return max(local)

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))

    return [word for word in tokens if word not in stop_words]
