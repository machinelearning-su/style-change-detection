import pandas as pd
from transformers import max_diff
from chunkers import word_chunks
from features import lexical, global_ngrams, readability, processed_tags
from utils import get_data
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from features.word_frequency import WordFrequency
from features.ngrams import NGrams
from preprocessing.basic_preprocessor import BasicPreprocessor
from keras.preprocessing import sequence

wf = WordFrequency()
preprocessor = BasicPreprocessor()

TRAINING_EXTERNAL_FILE = 'data/feather/external_feather'
TRAIN_X_FILE = 'data/feature_vectors/external_x.npy'
TRAIN_Y_FILE = 'data/feature_vectors/external_y.npy'

def main(cutoff=10000, persist=True):
    train_x, train_y, train_positions, train_file_names = get_data(
        external_file=TRAINING_EXTERNAL_FILE
    )

    if cutoff:
        train_x = train_x[:cutoff]
        train_y = train_y[:cutoff]
        train_positions = train_positions[:cutoff]

    pos_tag_x = [NGrams.to_pos_tags(x) for x in train_x]
    ngrams = NGrams(train_x, pos_tag_x)

    X = [preprocessor.process_text(x) for x in train_x]

    X_word_chunks = word_chunks(X, n=300, process=True)
    X_pos_chunks = word_chunks(pos_tag_x, n=300, process=True)

    max_segments = 10

    lexical_features = sequence.pad_sequences(lexical(X_word_chunks), maxlen=max_segments)
    stop_word_features = sequence.pad_sequences(ngrams.get_stop_words(X_word_chunks), maxlen=max_segments)
    function_word_features = sequence.pad_sequences(ngrams.get_function_words(X_word_chunks), maxlen=max_segments)
    pos_tag_features = sequence.pad_sequences(ngrams.get_pos_tags(X_pos_chunks), maxlen=max_segments)
    process_tag_features = sequence.pad_sequences(processed_tags(X_word_chunks), maxlen=max_segments)
    word_frequency = sequence.pad_sequences(wf.average_word_frequency(X_word_chunks), maxlen=max_segments)
    readability_features = sequence.pad_sequences(readability(X_word_chunks), maxlen=max_segments)

    print(type(lexical_features))

    X = np.concatenate([lexical_features, stop_word_features,
                            function_word_features, pos_tag_features, 
                            process_tag_features, word_frequency,
                            readability_features], axis=2)

    if persist:
        np.save(TRAIN_X_FILE, X)
        np.save(TRAIN_Y_FILE, train_y)
    

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    m, s = divmod(t_end - t_start, 60)
    print("All done in %fm and %fs" % (round(m), round(s)))
