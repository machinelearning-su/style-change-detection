from utils import print_progress_bar

def quote_discrepancies(data, feature_names=[]):
    with open('data/external/apostrophe_words.txt', 'r') as f:
        apostrophe_words = list(map(lambda x: x.split(',')[0], f.read().splitlines()))

    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        entry = entry.lower()

        single_quote_apostrophes = sum(map(lambda t: entry.count(t), apostrophe_words))

        count_single = entry.count("\'") - single_quote_apostrophes
        count_double = entry.count("\"")

        vectors.append([float(min(count_single, count_double))])

        print_progress_bar(i + 1, data_length, description = 'quote_discrepancies')

    feature_names.extend(['quote_discrepancies'])

    return vectors
