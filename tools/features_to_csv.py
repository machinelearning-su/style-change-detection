import pandas as pd
from transformers import max_diff
from chunkers import word_chunks
from features import lexical, global_ngrams, readability, processed_tags
from utils import get_data
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from features.word_frequency import WordFrequency
from features.ngrams import NGrams
from preprocessing.basic_preprocessor import BasicPreprocessor

wf = WordFrequency()
preprocessor = BasicPreprocessor()

TRAINING_DIR = 'data/training'
TRAIN_CSV_FILE = 'data/feature_vectors/train.csv'

def main(cutoff=None, persist=False):
    train_x, train_y, train_positions, train_file_names = get_data(
        main_dir=TRAINING_DIR
    )

    if cutoff:
        train_x = train_x[:cutoff]
        train_y = train_y[:cutoff]
        train_positions = train_positions[:cutoff]

    df = pd.DataFrame(data={'label': train_y, 'pos': train_positions})    

    pos_tag_x = [NGrams.to_pos_tags(x) for x in train_x]
    ngrams = NGrams(train_x, pos_tag_x)

    X = [preprocessor.process_text(x) for x in train_x]

    X_word_chunks = word_chunks(X, n=300, process=True)
    X_pos_chunks = word_chunks(pos_tag_x, n=300, process=True)

    fmap = {
        'lexical_features': lexical(X_word_chunks),
        'stop_word_features': ngrams.get_stop_words(X_word_chunks),
        'function_word_features': ngrams.get_function_words(X_word_chunks),
        'pos_tag_features': ngrams.get_pos_tags(X_pos_chunks),
        'process_tag_features': processed_tags(X_word_chunks),
        'word_frequency': wf.average_word_frequency(X_word_chunks),
        'readability_features': readability(X_word_chunks)
    }

    for key, feature in fmap.items():
        df[key] = feature
    

    if persist:
        df.to_csv(TRAIN_CSV_FILE)
    

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    m, s = divmod(t_end - t_start, 60)
    print("All done in %fm and %fs" % (round(m), round(s)))
