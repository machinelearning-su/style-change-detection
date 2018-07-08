import math, re
from operator import itemgetter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from utils import print_progress_bar

def split_points(data, word_weights):
    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        entry = entry.lower()

        words = word_tokenize(entry)

        sum_vector = 0

        for word in words:
            if(word in word_weights):
                sum_vector += word_weights[word]

        vectors.append([sum_vector])

        print_progress_bar(i + 1, data_length, description = 'split_points')

    return vectors

def calculate_weights(data, train_positions, inverse_scaling, half_sigmoid_sharpness, size):
    word_weights = {}
    word_counts = {}

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

            fragment_length = len(words)

            for position, word in enumerate(words):
                if(size and position >= size and position <= fragment_length - size - 1):
                    continue

                word_weight = weight_half_sigmoid(position, fragment_length, half_sigmoid_sharpness)
                if word in word_weights:
                    word_weights[word].append(word_weight)
                    word_counts[word] = word_counts[word] + 1
                else:
                    word_weights[word] = [word_weight]
                    word_counts[word] = 1

        print_progress_bar(i + 1, data_length, description = 'split_points_weights')

    remove_entries(word_weights, stopwords.words('english'))
    remove_entries(word_counts, stopwords.words('english'))

    max_word_count = word_counts[max(word_counts.items(), key=itemgetter(1))[0]]
    min_word_count = word_counts[min(word_counts.items(), key=itemgetter(1))[0]]

    if(inverse_scaling):
        additional_weight = lambda k: float((word_counts[k] + 1 - min_word_count) / (max_word_count + 1 - min_word_count))

        word_weights = {k: (sum(v) / float(len(v))) * additional_weight(k) for k, v in word_weights.items()}
    else:
        word_weights = {k: sum(v) / float(len(v)) for k, v in word_weights.items()}

    for key, value in sorted(word_weights.items(), key = itemgetter(1), reverse = True)[:50]: print(key, value)

    return word_weights

def remove_entries(dictionary, entries):
    for entry in entries:
        dictionary.pop(entry, None)

def weight_sigmoid(position, fragment_length):
    x = float(min(float(position / fragment_length), 1 - float(position / fragment_length)))

    exponent = 7 * ((2*x) - 1)

    return float(1 / (1 + math.exp(exponent)))

# k - defines the "sharpness" of the function
def weight_half_sigmoid(position, fragment_length, k):
    half = float(fragment_length / 2)

    x = float(abs(half - (position + 1)) / half)

    return float(((0 + k) * x) / ((1 + k) - x))
