import numpy as np
import pandas as pd
import nltk
import tqdm

nltk.download('punkt')
nltk.download('wordnet')

from keras.callbacks import ModelCheckpoint, EarlyStopping

import re, os, gc, time, pandas as pd, numpy as np


from nltk import tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, Add, Flatten, TimeDistributed,CuDNNGRU,CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
# from keras.engine.topology import Layer
from keras.engine import InputSpec, Layer

from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras import constraints

BATCH_SIZE = 128
DENSE_SIZE = 32
RECURRENT_SIZE = 64
DROPOUT_RATE = 0.3
MAX_SENTENCE_LENGTH = 500
OUTPUT_CLASSES = 1
MAX_EPOCHS = 18

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["is_multi_authir"]
embed_size = 300
max_features = 150000
max_text_len = 200
max_sent = 10

print("Loading data...")

train = pd.read_csv("/pan_data/train.csv")
test = pd.read_csv("/pan_data/test.csv")
test_ids = test[['id']].values

train["comment_text"].fillna("no comment", inplace = True)
test["comment_text"].fillna("no comment", inplace = True)

train["sentences"] = train["comment_text"].apply(lambda x: tokenize.sent_tokenize(x))
test["sentences"] = test["comment_text"].apply(lambda x: tokenize.sent_tokenize(x))

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

raw_text = train["comment_text"]
tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text)

y_train = train[CLASSES].values

def sentenize(data):
    comments = data["sentences"]
    sent_matrix = np.zeros((len(comments), max_sent, max_text_len), dtype = "int32")
    for i, sentences in enumerate(comments):
        for j, sent in enumerate(sentences):
            if j < max_sent:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_text_len and tk.word_index[word] < max_features:
                            sent_matrix[i, j, k] = tk.word_index[word]
                            k = k+1
                    except:
                            sent_matrix[i, j, k] = 0
                            k = k+1
    return sent_matrix


X = sentenize(train)
X_test = sentenize(test)

del train, test
gc.collect()

EMBEDDING_FILE = "/pan_data/crawl-300d-2M.vec"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype = "float32")
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))



word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        # ait = K.dot(uit, self.u)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

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
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
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

print("Starting to train models...")

fold = 0
test_predicts_list = []

fold_size = len(X) // 10
total_meta = []
fold_count = 10

rnn_units = 72
de_units = 72

for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_count - 1:
        fold_end = len(X)

    train_x = np.concatenate([X[:fold_start], X[fold_end:]])
    train_y = np.concatenate([y_train[:fold_start], y_train[fold_end:]])

    val_x = X[fold_start:fold_end]
    val_y = y_train[fold_start:fold_end]

    encoder_inp = Input(shape=(max_text_len,), dtype="int32")
    endcoder = Embedding(nb_words, embed_size, weights=[embedding_matrix],
                         input_length=max_text_len, trainable=False)(encoder_inp)
    endcoder = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True))(endcoder)
    endcoder = TimeDistributed(Dense(de_units, activation="relu"))(endcoder)
    endcoder = AttentionWeightedAverage()(endcoder)
    Encoder = Model(encoder_inp, endcoder)

    decoder_inp = Input(shape=(max_sent, max_text_len), dtype="int32")
    decoder = TimeDistributed(Encoder)(decoder_inp)
    decoder = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True))(decoder)
    decoder = TimeDistributed(Dense(de_units, activation="relu"))(decoder)
    Decoder = AttentionWeightedAverage()(decoder)
    Decoder = Dropout(0.3)(Decoder)
    # Decoder = Dropout(0.7)(Decoder)
    out = Dense(1, activation="sigmoid")(Decoder)
    model = Model(decoder_inp, out)

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

    # train the model
    history = model.fit(train_x,
                        train_y,
                        epochs=500,
                        batch_size=128,
                        validation_data=(val_x, val_y),
                        callbacks=[earlystopper, checkpointer],
                        shuffle=True)

    model.load_weights(filepath="weights.fold." + str(fold) + ".hdf5", by_name=False)

    predictions = model.predict(X_test, batch_size=256)
    test_predicts_list.append(predictions)

    meta = model.predict(val_x, batch_size=256)
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
test_predicts.to_csv("/output/HAN_crawl_sub.csv", index=False)

print("Making meta predictions...")

subm = pd.read_csv("/pan_data/train.csv")
submid = pd.DataFrame({'id': subm["id"]})
total_meta_data = pd.concat([submid, pd.DataFrame(total_meta, columns=CLASSES)], axis=1)
total_meta_data.to_csv('/output/HAN_crawl_meta.csv', index = False)
print("Meta predicted !!!!")
