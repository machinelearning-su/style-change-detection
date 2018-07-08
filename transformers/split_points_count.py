import math, re
from operator import itemgetter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from utils import print_progress_bar

def split_points_count(data, words_left, words_right, window_words):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        entry = entry.lower()

        words = word_tokenize(entry)

        local = []

        index = 0
        while(index <= len(words) - window_words):
            summ = 0
            for word in words[index:index + window_words]:
                if(word in words_left): l = words_left[word]
                else: l = 0
                if(word in words_right): r = words_right[word]
                else: r = 0

                summ += max(l, r)
            index += window_words
            local.append(summ)

        vectors.append([max(local)])
        print_progress_bar(i + 1, data_length, description = 'split_points')

    return vectors

def calculate_weights_count(data, train_positions, inverse_scaling, half_sigmoid_sharpness, size):
    words_left = {}
    words_right = {}
    words_global = {}

    data_length = len(data)

    for i, (entry, positions) in enumerate(zip(data, train_positions)):
        entry = entry.lower()

        fragments = []
        positions.append(len(entry))

        entry_marker = 0

        for change in positions:
            fragments.append(entry[entry_marker:change])
            entry_marker = change

        for fragment in fragments:
            fragment = re.sub("[^a-zA-Z]+", " ", fragment)
            words = word_tokenize(fragment)
            left = words[:size]
            right = words[-size:]

            for word in words:
                if word in words_global: words_global[word] += 1
                else: words_global[word] = 1

            for word in left:
                if word in words_left: words_left[word] = words_left[word] + 1
                else: words_left[word] = 1

            for word in right:
                if word in words_right: words_right[word] = words_right[word] + 1
                else: words_right[word] = 1

        print_progress_bar(i + 1, data_length, description = 'split_points_weights')

    remove_entries(words_left, stopwords.words('english'))
    remove_entries(words_right, stopwords.words('english'))

    words_left = min_max_dict(words_left)
    words_right = min_max_dict(words_right)

    for key, value in sorted(words_left.items(), key = itemgetter(1), reverse = True)[:50]: print(key, value)
    print('====================================================')
    for key, value in sorted(words_right.items(), key = itemgetter(1), reverse = True)[:50]: print(key, value)

    return words_left, words_right

def min_max_dict(dictionary):
    max_count = dictionary[max(dictionary.items(), key=itemgetter(1))[0]]
    min_count = dictionary[min(dictionary.items(), key=itemgetter(1))[0]]

    return {k: float((v - min_count) / (max_count - min_count)) for k, v in dictionary.items()}

def remove_entries(dictionary, entries):
    for entry in entries:
        dictionary.pop(entry, None)
