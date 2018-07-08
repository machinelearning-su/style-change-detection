from textstat.textstat import textstat
from utils import print_progress_bar

def readability(X, indices = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index', \
     'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', \
     'linsear_write_formula', 'gunning_fog'], feature_names = []):
    transformed = []

    data_length = len(X)

    for i, doc in enumerate(X):
        segments = []

        for entry in doc:
            segments.append([getattr(textstat, index)(entry) for index in indices])

        transformed.append(segments)

        print_progress_bar(i + 1, data_length, description = 'readability')

    feature_names.extend(indices)

    return transformed
