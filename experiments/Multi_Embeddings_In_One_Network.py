import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.layers import Dense, Embedding, Input, \
    Bidirectional, Dropout, CuDNNGRU, GRU, \
    Input, Activation, LSTM, GlobalMaxPool1D, BatchNormalization, GlobalMaxPooling1D, \
    Convolution1D, Conv1D, InputSpec, Flatten, GlobalAveragePooling1D, Concatenate, TimeDistributed, Lambda, CuDNNLSTM, \
    ZeroPadding1D


from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

import numpy as np
import re
import pandas as pd

from keras import backend as K

BATCH_SIZE = 128
DENSE_SIZE = 32
RECURRENT_SIZE = 64
DROPOUT_RATE = 0.3
MAX_SENTENCE_LENGTH = 500
OUTPUT_CLASSES = 6
MAX_EPOCHS = 18

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print("Loading data...")

train = pd.read_csv("/camera_data/train_clean.csv")
test = pd.read_csv("/camera_data/test_clean.csv")
test_ids = test[['id']].values

list_sentences_train = train["comment_text"].fillna(NAN_WORD)
list_sentences_test = test["comment_text"].fillna(NAN_WORD)

list_sentences_train = list_sentences_train.values
list_sentences_test = list_sentences_test.values
y_train = train[CLASSES].values


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train


print("Tokenizing train data")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
tokenized_sentences_train_glove, words_dict_glove = tokenize_sentences(list_sentences_train, {})

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)
tokenized_sentences_test_glove, words_dict_glove = tokenize_sentences(list_sentences_test, words_dict_glove)

print("Loading Crawl Embeddings...")
embedding_list, embedding_word_dict = read_embedding_list("/pan_data/crawl-300d-2M.vec")
embedding_size = len(embedding_list[0])

print("Loading Glove embeddings...")
embedding_list_glove, embedding_word_dict_glove = read_embedding_list("/pan_data/glove.840B.300d.txt")
embedding_size_glove = len(embedding_list_glove[0])

print("Preparing data...")

embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)
embedding_list_glove, embedding_word_dict_glove = clear_embedding_list(embedding_list_glove, embedding_word_dict_glove,
                                                                       words_dict_glove)

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_word_dict_glove[UNKNOWN_WORD] = len(embedding_word_dict_glove)
embedding_list_glove.append([0.] * embedding_size_glove)
embedding_word_dict_glove[END_WORD] = len(embedding_word_dict_glove)
embedding_list_glove.append([-1.] * embedding_size_glove)

embedding_matrix = np.array(embedding_list)
embedding_matrix_glove = np.array(embedding_list_glove)

id_to_word = dict((id, word) for word, id in words_dict.items())
id_to_word_glove = dict((id, word) for word, id in words_dict_glove.items())

train_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    MAX_SENTENCE_LENGTH)

test_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    MAX_SENTENCE_LENGTH)

train_list_of_token_ids_glove = convert_tokens_to_ids(
    tokenized_sentences_train_glove,
    id_to_word_glove,
    embedding_word_dict_glove,
    MAX_SENTENCE_LENGTH)

test_list_of_token_ids_glove = convert_tokens_to_ids(
    tokenized_sentences_test_glove,
    id_to_word_glove,
    embedding_word_dict_glove,
    MAX_SENTENCE_LENGTH)

X_train = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)

X_train_glove = np.array(train_list_of_token_ids_glove)
X_test_glove = np.array(test_list_of_token_ids_glove)

print("Starting to train models...")

fold = 0
test_predicts_list = []

fold_size = len(X_train) // 10
total_meta = []
fold_count = 10

for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X_train)

    train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
    train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x = X_train[fold_start:fold_end]
    val_y = y_train[fold_start:fold_end]

    train_x_glove = np.concatenate([X_train_glove[:fold_start], X_train_glove[fold_end:]])
    train_y_glove = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x_glove = X_train_glove[fold_start:fold_end]
    val_y_glove = y_train[fold_start:fold_end]

    input_crawl = Input(shape=(MAX_SENTENCE_LENGTH,), name='crawl_input')
    input_glove = Input(shape=(MAX_SENTENCE_LENGTH,), name='glove_input')

    crawl_embeddings = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                                 weights=[embedding_matrix], trainable=False)(input_crawl)

    crawlDropout = SpatialDropout1D(0.14893892839382)(crawl_embeddings)

    glove_embeddings = Embedding(embedding_matrix_glove.shape[0], (embedding_matrix_glove.shape[1]),
                                 weights=[embedding_matrix_glove], trainable=False)(input_glove)
    gloveDropout = SpatialDropout1D(0.138938382322)(glove_embeddings)

    crawlGRU = Bidirectional(CuDNNGRU(64, return_sequences=True))(crawlDropout)
    gloveLSTM = Bidirectional(CuDNNLSTM(64, return_sequences=True))(gloveDropout)

    filter_sizes = [3, 5, 7]

    conv_pools = []

    for filter_size in filter_sizes:
        l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(crawlGRU)
        l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='same', activation='elu')(l_zero)
        l_pool = GlobalMaxPool1D()(l_conv)
        conv_pools.append(l_pool)

    for filter_size in filter_sizes:
        l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(gloveLSTM)
        l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='same', activation='elu')(l_zero)
        l_pool = GlobalMaxPool1D()(l_conv)
        conv_pools.append(l_pool)

    l_merge = Concatenate(axis=1)(conv_pools)

    l_dense = Dense(128, activation='elu')(l_merge)
    l_dropout = Dropout(0.1134837438472)(l_dense)
    l_out = Dense(1, activation='sigmoid')(l_dense)

    model = Model(inputs=[input_crawl, input_glove], output=l_out)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['acc'])

    checkpointer = ModelCheckpoint(filepath="weights.fold." + str(fold) + ".hdf5",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   verbose=1)

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # train the model
    history = model.fit([train_x, train_x_glove],
                        train_y,
                        epochs=50,
                        batch_size=256,
                        validation_data=([val_x, val_x_glove], val_y),
                        callbacks=[earlystopper, checkpointer],
                        shuffle=True)

    model.load_weights(filepath="weights.fold." + str(fold) + ".hdf5", by_name=False)

    predictions = model.predict([X_test, X_test_glove], batch_size=256)

    test_predicts_list.append(predictions)

    meta = model.predict([val_x, val_x_glove], batch_size=256)

    if (fold_id == 0):
        total_meta = meta
    else:
        total_meta = np.concatenate((total_meta, meta), axis=0)

    K.clear_session()

print("Making predictions")
test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))

test_ids = test_ids.reshape((len(test_ids), 1))
test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
test_predicts.to_csv("/output/multi_embeddings.csv", index=False)

print("Making meta predictions...")

subm = pd.read_csv("/pan_data/train_clean.csv")
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns=CLASSES)], axis=1)
total_meta_data.to_csv('/output/multi_embeddingsd_meta.csv', index=False)
print("Meta predicted !!!!")