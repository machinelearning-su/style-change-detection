import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.layers import Dense, Embedding, Input, \
    Bidirectional, Dropout, CuDNNGRU, CuDNNLSTM, PReLU, GRU, \
    Input, Activation, LSTM, GlobalMaxPool1D, BatchNormalization, GlobalMaxPooling1D, \
    Convolution1D, Conv1D, InputSpec, Flatten, GlobalAveragePooling1D, Concatenate, TimeDistributed, Lambda

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

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
act, pad, kernel_ini = "linear", "same", "he_uniform"

for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X_train)

    train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
    train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x = X_train[fold_start:fold_end]
    val_y = y_train[fold_start:fold_end]

    # inp = Input(shape=(500,))
    # emb  = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
    #                            weights=[embedding_matrix], trainable=False)(inp)
    # x = Bidirectional(CuDNNGRU(gru_len,return_sequences=True))(emb)
    # x = SpatialDropout1D(0.119039436667)(x)
    #  x = Activation('elu')(x)
    #  emb = SpatialDropout1D(0.2)(x)
    #   capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,share_weights=True)(x)
    # capsule = Capsule(num_capsule=128, dim_capsule=32, routings=3)(emb)
    #    capsule = Capsule(num_capsule=64, dim_capsule=32, routings=3)(capsule)
    #   capsule = Capsule(num_capsule=32, dim_capsule=32, routings=3)(capsule)
    #  capsule = Capsule(num_capsule=16, dim_capsule=32, routings=3)(capsule)
    # capsule = Capsule(num_capsule=8, dim_capsule=32, routings=3)(capsule)

    #  capsule = Flatten()(capsule)

    # model = Model(inputs=inp, outputs=capsule)

    input = Input(shape=(500,))

    embedding_layer = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                                weights=[embedding_matrix], trainable=False)(input)
    emb = SpatialDropout1D(0.2)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(gru_len,return_sequences=True))(embedding_layer)
    # x = SpatialDropout1D(0.119039436667)(x)
    # x = Activation('elu')(x)

    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,share_weights=True)(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(emb)
    x = SpatialDropout1D(0.119039436667)(x)
    x = Activation('elu')(x)

    capsule = Capsule(num_capsule=128, dim_capsule=32, routings=3)(x)
    capsule = Capsule(num_capsule=64, dim_capsule=32, routings=3)(capsule)
    capsule = Capsule(num_capsule=32, dim_capsule=32, routings=3)(capsule)
    capsule = Capsule(num_capsule=16, dim_capsule=32, routings=3)(capsule)
    capsule = Capsule(num_capsule=8, dim_capsule=32, routings=3)(capsule)

    capsule = Flatten()(capsule)

    capsule = Dropout(dropout_p)(capsule)

    output_layer = Dense(1, activation="sigmoid")(capsule)

    model = Model(inputs=input, outputs=output_layer)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="weights.fold." + str(fold) + ".hdf5",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   verbose=1)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # train the model
    history = model.fit(train_x,
                        train_y,
                        epochs=500,
                        batch_size=128,
                        validation_data=(val_x, val_y),
                        callbacks=[earlystopper, checkpointer],
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
test_predicts.to_csv("/output/deep_capsule_sub.csv", index=False)

print("Making meta predictions...")

subm = pd.read_csv("/pan_data/train.csv")
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns=CLASSES)], axis=1)
total_meta_data.to_csv('/output/deep_capsule_meta.csv', index = False)
print("Meta predicted !!!!")
