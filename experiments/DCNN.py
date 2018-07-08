
import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')

from keras.layers import Dense, Embedding, Input, \
    Bidirectional, Dropout, CuDNNGRU, GRU, \
    Input, Activation, LSTM, GlobalMaxPool1D, BatchNormalization, GlobalMaxPooling1D, \
    Convolution1D, Conv1D, InputSpec, Flatten, GlobalAveragePooling1D, Concatenate, TimeDistributed, Lambda
from keras.layers.merge import add, concatenate

from keras.activations import relu

from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU, Input, Convolution1D, GlobalMaxPooling1D, \
    PReLU
from keras.layers.normalization import BatchNormalization

from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints


import numpy as np
import re
import pandas as pd

BATCH_SIZE = 128
DENSE_SIZE = 32
RECURRENT_SIZE = 64
DROPOUT_RATE = 0.3
MAX_SENTENCE_LENGTH = 500
OUTPUT_CLASSES = 1
MAX_EPOCHS = 8

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


def _dropout(dropout, dropout_mode):
    def f(x):
        if dropout_mode == 'spatial':
            x = SpatialDropout1D(dropout)(x)
        elif dropout_mode == 'simple':
            x = Dropout(dropout)(x)
        else:
            raise NotImplementedError('spatial and simple modes are supported')
        return x

    return f

def _prelu(use_prelu):
    def f(x):
        if use_prelu:
            x = PReLU()(x)
        else:
            x = Lambda(relu)(x)
        return x

    return f


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / K.sum(ai, axis=1, keepdims=True)
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def _vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg, last_block):
    def f(x):
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        x = add([main, x])
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        x = add([main, x])
        if not last_block:
            x = MaxPooling1D(pool_size=3, strides=2)(x)
        return x

    return f


def _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                         kernel_reg_l2, bias_reg_l2, batch_norm_first):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same', activation='linear',
                   kernel_regularizer=regularizers.l2(kernel_reg_l2),
                   bias_regularizer=regularizers.l2(bias_reg_l2))(x)
        x = _bn_relu_dropout_block(use_batch_norm=use_batch_norm,
                                   batch_norm_first=batch_norm_first,
                                   dropout=dropout,
                                   dropout_mode=dropout_mode,
                                   use_prelu=use_prelu)(x)
        return x

    return f


def _dense_block(dense_size, use_batch_norm, use_prelu, dropout, kernel_reg_l2, bias_reg_l2,
                 batch_norm_first):
    def f(x):
        x = Dense(dense_size, activation='linear',
                  kernel_regularizer=regularizers.l2(kernel_reg_l2),
                  bias_regularizer=regularizers.l2(bias_reg_l2))(x)

        x = _bn_relu_dropout_block(use_batch_norm=use_batch_norm,
                                   use_prelu=use_prelu,
                                   dropout=dropout,
                                   dropout_mode='simple',
                                   batch_norm_first=batch_norm_first)(x)
        return x

    return f

def _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout, dropout_mode, batch_norm_first):
    def f(x):
        if use_batch_norm and batch_norm_first:
            x = BatchNormalization()(x)

        x = _prelu(use_prelu)(x)
        x = _dropout(dropout, dropout_mode)(x)

        if use_batch_norm and not batch_norm_first:
            x = BatchNormalization()(x)
        return x

    return f

def _shape_matching_layer(filter_nr, use_prelu, kernel_reg_l2, bias_reg_l2):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                   kernel_regularizer=regularizers.l2(kernel_reg_l2),
                   bias_regularizer=regularizers.l2(bias_reg_l2))(x)
        x = _prelu(use_prelu)(x)
        return x

    return f

def _classification_block(dense_size, repeat_dense,
                          max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                          dropout,
                          kernel_reg_l2, bias_reg_l2,
                          use_prelu, use_batch_norm, batch_norm_first):
    def f(x):
        if max_pooling:
            x_max = GlobalMaxPool1D()(x)
        else:
            x_max = None

        if mean_pooling:
            x_mean = GlobalAveragePooling1D()(x)
        else:
            x_mean = None
        if weighted_average_attention:
            x_att = AttentionWeightedAverage()(x)
        else:
            x_att = None

        x = [xi for xi in [x_max, x_mean, x_att] if xi is not None]
        if len(x) == 1:
            x = x[0]
        else:
            if concat_mode == 'concat':
                x = concatenate(x, axis=-1)
            else:
                NotImplementedError('only mode concat for now')

        for _ in range(repeat_dense):
            x = _dense_block(dense_size=dense_size,
                             use_batch_norm=use_batch_norm,
                             use_prelu=use_prelu,
                             dropout=dropout,
                             kernel_reg_l2=kernel_reg_l2,
                             bias_reg_l2=bias_reg_l2,
                             batch_norm_first=batch_norm_first)(x)

        x = Dense(1, activation="sigmoid")(x)
        return x

    return f


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


word_embedding_size = (embedding_matrix.shape[1])

batch_norm_first = 1
use_batch_norm=  1
dropout_embedding = 0.5
rnn_dropout=  0.5
dense_dropout=  0.5
conv_dropout=  0.2
dropout_mode = 'spatial'
rnn_kernel_reg_l2 = None
rnn_recurrent_reg_l2 = None
rnn_bias_reg_l2=  None
dense_kernel_reg_l2 = 0.00001
dense_bias_reg_l2 = 0.00001
conv_kernel_reg_l2 =  0.00001
conv_bias_reg_l2 =  0.00001

epochs_nr = 1000
batch_size_train = 128
batch_size_inference = 128
lr = 0.005
momentum = 0.9
gamma = 0.8
patience=  5

filter_nr = 64
kernel_size=  3
repeat_block = 3
dense_size = 256
repeat_dense = 1
max_pooling = 1
mean_pooling = 0
weighted_average_attention = 0
concat_mode = 'concat'
trainable_embedding = 0
word_embedding_size = 300
char_embedding_size = None

l2_reg_convo = 0.00001
dropout_convo = 0.2

use_prelu = 1
fold = 0


def _dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                 kernel_reg_l2, bias_reg_l2, batch_norm_first):
    def f(x):
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                    kernel_reg_l2, bias_reg_l2, batch_norm_first)(x)
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, dropout_mode,
                                    kernel_reg_l2, bias_reg_l2, batch_norm_first)(main)
        x = add([main, x])
        return x

    return f

for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X_train)

    train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
    train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x = X_train[fold_start:fold_end]
    val_y = y_train[fold_start:fold_end]


    input_text = Input(shape=(MAX_SENTENCE_LENGTH,))

    embedding = Embedding(embedding_matrix.shape[0], (embedding_matrix.shape[1]),
                          weights=[embedding_matrix], trainable=False)(input_text)

    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                             conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(embedding)
    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                             conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)

    if embedding_size == filter_nr:
        x = add([embedding, x])
    else:
        embedding_resized = _shape_matching_layer(filter_nr, use_prelu, conv_kernel_reg_l2, conv_bias_reg_l2)(embedding)
        x = add([embedding_resized, x])

    for _ in range(repeat_block):
        x = _dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, conv_dropout, dropout_mode,
                         conv_kernel_reg_l2, conv_bias_reg_l2, batch_norm_first)(x)

    predictions = _classification_block(dense_size=dense_size, repeat_dense=repeat_dense,
                                        max_pooling=max_pooling,
                                        mean_pooling=mean_pooling,
                                        weighted_average_attention=weighted_average_attention,
                                        concat_mode=concat_mode,
                                        dropout=dense_dropout,
                                        kernel_reg_l2=dense_kernel_reg_l2, bias_reg_l2=dense_bias_reg_l2,
                                        use_prelu=use_prelu, use_batch_norm=use_batch_norm,
                                        batch_norm_first=batch_norm_first)(x)

    model = Model(inputs=input_text, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="weights.fold." + str(fold) + ".hdf5",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   monitor='val_loss',
                                   verbose=1)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    # train the model
    history = model.fit(train_x,
                        train_y,
                        epochs=50,
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
# test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

test_ids = test_ids.reshape((len(test_ids), 1))
test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
test_predicts.to_csv("/output/script_dpcnn_crawl.csv", index=False)

print("Making meta predictions...")

subm = pd.read_csv("/pan_data/train.csv")
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns=CLASSES)], axis=1)
total_meta_data.to_csv('/output/dpcnn_crawl_meta.csv', index = False)
print("Meta predicted !!!!")