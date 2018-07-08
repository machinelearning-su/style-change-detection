import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.layers import Dense, Embedding, Input, \
    Bidirectional, Dropout, CuDNNGRU, CuDNNLSTM, PReLU, GRU, \
    Input, Activation, LSTM, GlobalMaxPool1D, BatchNormalization, GlobalMaxPooling1D, \
    Convolution1D, Conv1D, InputSpec, Flatten, GlobalAveragePooling1D, Concatenate, TimeDistributed, Lambda, Conv2D, MaxPool2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,GlobalMaxPool2D, GlobalMaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers import merge

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K


from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

import numpy as np
import pandas as pd
import re
import random
import pandas as pd
import requests


BATCH_SIZE = 256
DENSE_SIZE = 32
RECURRENT_SIZE = 64
DROPOUT_RATE = 0.3
MAX_SENTENCE_LENGTH = 500
OUTPUT_CLASSES = 1
MAX_EPOCHS = 18

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["is_multi_author"]
print("Loading data...")

train = pd.read_csv("/pan_data/train.csv")
test = pd.read_csv("/pan_data/test.csv")
test_ids = test[['id']].values

list_sentences_train = train["comment_text"].fillna(NAN_WORD)
list_sentences_test = test["comment_text"].fillna(NAN_WORD)

list_sentences_train = list_sentences_train.values
list_sentences_test = list_sentences_test.values
y_train = train[CLASSES].values

from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras import constraints

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

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list("/camera_data2/crawl-300d-2M.vec")
embedding_size = len(embedding_list[0])

print("Preparing data...")

embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())

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

X_train = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)

print("Starting to train models...")

fold = 0
test_predicts_list = []

fold_size = len(X_train) // 10
total_meta = []
fold_count = 10

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28


def schedule(ind):
    a = [0.001, 0.0005, 0.00025, 0.000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125,0.0000125]
    return a[ind]

filter_sizes = [3,5]

num_filters = 32

window_sizes=[3,5]

for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X_train)

    train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
    train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x = X_train[fold_start:fold_end]
    val_y = y_train[fold_start:fold_end]

    input = Input(shape=(500,))

    channel_finetune = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                                weights=[embedding_matrix], trainable=False, name='FineTune_Channel')(input)
    channel_static = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                                weights=[embedding_matrix], trainable=True, name='Static_Channel')(input)

    reshape_layer_finetune = Reshape(target_shape=(1, 500, (embedding_matrix.shape[1])), name='Reshape_Layer_FineTune')(channel_finetune)
    reshape_layer_static = Reshape(target_shape=(1, 500, (embedding_matrix.shape[1])), name='Reshape_Layer_Static')(channel_static)

    channels = [reshape_layer_finetune, reshape_layer_static]
    filter_varying_prop = []

    for size in window_sizes:
        conv_value_list = []
        for i, channel in enumerate(channels):
            if (i == 0):
                val = '_FINETUNE'
            else:
                val = '_STATIC'
            conv_layer = Conv2D(filters=128, kernel_size=(size, 300), strides=(1, 1), padding='valid',
                                data_format='channels_first', activation='selu',
                                name='CONV_FOR_WINDOW_' + str(size) + val)(channel)
            conv_value_list.append(conv_layer)
        multichannel_adding_layer = merge(conv_value_list, mode='sum', name='MULTICHANNEL_ADDING_LAYER_' + str(size))
        pooling_layer = GlobalMaxPool2D(data_format='channels_first', name='POOL_FOR_WINDOW_' + str(size))(multichannel_adding_layer)
        filter_varying_prop.append(pooling_layer)
    concatenate_layer = merge(filter_varying_prop, mode='concat', concat_axis=-1, name='Concatenate_Layer')

    concatenate_layer = Dropout(0.25)(concatenate_layer)

    regulator = Dense(64, activation='elu')(concatenate_layer)

    regulator = Dropout(0.10239029039203923902)(regulator)

    output_layer = Dense(1, activation="sigmoid")(regulator)

    model = Model(inputs=input, outputs=output_layer)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="weights.fold." + str(fold) + ".hdf5",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   verbose=1)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    lr = LearningRateScheduler(schedule)

    # train the model
    history = model.fit(train_x,
                        train_y,
                        epochs=200,
                        batch_size=128,
                        validation_data=(val_x, val_y),
                        callbacks=[earlystopper, checkpointer, lr],
                        shuffle=True)

    model.load_weights(filepath="weights.fold." + str(fold) + ".hdf5", by_name=False)

    predictions = model.predict(X_test, batch_size=128)
    test_predicts_list.append(predictions)

    meta = model.predict(val_x, batch_size=128)
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
test_predicts.to_csv("/output/conv2d_multi_channel_sub.csv", index=False)

print("Making meta predictions...")

subm = pd.read_csv("/pan_data/train.csv")
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns=CLASSES)], axis=1)
total_meta_data.to_csv('/output/conv2d_multi_channel_meta.csv', index = False)
print("Meta predicted !!!!")