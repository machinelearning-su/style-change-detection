import pandas as pd
import lightgbm as lgb # pip install lightgbm to get this shit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import  StratifiedKFold # always go #Stratified on Twitter


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

CLASS_NAMES = ["is_multi_author"]
TARGET_LABEL = "is_multi_author"
OFFICIAL_PAN_TRAIN_FILE = "my_train.csv"
OFFICIAL_PAN_VALIDATION_FILE = "my_test.csv" # Used for final prediction, it is not involved in the validation
TEXT_AS_KEY = "text"
MAX_FEATURES = 300000 # Used at both word and char level
SUCCESS_MESSAGE = "Done"
SUCCESS_HSTACK = "HStack Done"


LIGHTGBM_PARAMS =  {
        'learning_rate': 0.1,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc', # All our validation and calculation metrics are in terms of Accuracy ! I am adding another point of view with 'auc' for additional monitoring
        'data_random_seed': 2,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1,
}

# Despite having strong regularization policies with LightGBM, I am using strong policies with LogisticClassifier too
INVERSE_LOGISTIC_REGRESSION_STRENGTH = 2.0
LOGISTIC_SOLVER = 'sag'
MINIMUM_FEATURE_WEIGHT = 0.3

LIGHTGBM_OPTIMAL_ROUNDS = {
    'is_multi_author': 140
}

train = pd.read_csv(OFFICIAL_PAN_TRAIN_FILE).fillna(' ') # just in case, haven't checked that
test = pd.read_csv(OFFICIAL_PAN_VALIDATION_FILE).fillna(' ')
verification = pd.read_csv(OFFICIAL_PAN_TRAIN_FILE).fillna(' ') # same as train, used for comparison to calculate individual fold validation accuracy

# Fed these to the vectorizers
train_text = train[TEXT_AS_KEY]
test_text = test[TEXT_AS_KEY]

all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=MAX_FEATURES
)

word_vectorizer.fit(all_text)
print(SUCCESS_MESSAGE)
train_word_features = word_vectorizer.transform(train_text)
print(SUCCESS_MESSAGE)
test_word_features = word_vectorizer.transform(test_text)
print(SUCCESS_MESSAGE)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=MAX_FEATURES
)

char_vectorizer.fit(all_text)
print(SUCCESS_MESSAGE)
train_char_features = char_vectorizer.transform(train_text)
print(SUCCESS_MESSAGE)
test_char_features = char_vectorizer.transform(test_text)
print(SUCCESS_MESSAGE)

train_features = hstack([train_char_features, train_word_features])
print(SUCCESS_HSTACK)
test_features = hstack([test_char_features, test_word_features])
print(SUCCESS_HSTACK)

submission = pd.DataFrame() # Our final prediction will be stored here

train.drop('text', axis=1, inplace=True) # we do not need text anymore, sparse matrixes as features will cure us :)

del test
del train_text
del test_text
del all_text
del train_char_features
del test_char_features
del train_word_features
del test_word_features

gc.collect()

print("Mediocre Memory Optimization Done")

skf = StratifiedKFold(n_splits= 5, random_state= 777) # Yeah, hope to be lucky today

cnt = 0 # Used as a counter for our folds
cvAccuracy = 0 # Our total CV Accuracy for Bagging=5 will be stored here

for class_name in CLASS_NAMES:

    train_target = train[class_name]
    model = LogisticRegression(C=INVERSE_LOGISTIC_REGRESSION_STRENGTH, solver=LOGISTIC_SOLVER)
    sfm = SelectFromModel(model, threshold=MINIMUM_FEATURE_WEIGHT)

    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    print(train_sparse_matrix.shape)
    test_sparse_matrix = sfm.transform(test_features)
    print(test_sparse_matrix.shape)
    validation_sparse_matrix = sfm.transform(train_features)

    for (f, (train_index, validation_index)) in enumerate(skf.split(train_sparse_matrix, train_target)):
        cnt += 1
        d_train = lgb.Dataset(train_sparse_matrix[train_index], label=train_target[train_index])
        d_valid = lgb.Dataset(train_sparse_matrix[validation_index], label=train_target[validation_index])

        watchlist = [d_train, d_valid]

        model = lgb.train(LIGHTGBM_PARAMS,
                          train_set=d_train,
                          num_boost_round=LIGHTGBM_OPTIMAL_ROUNDS[class_name],
                          valid_sets=watchlist,
                          verbose_eval=10 # Ten is suitable choice for good monitoring
                          )

        crossValidationPrediction = pd.DataFrame()

        crossValidationPrediction[class_name] = model.predict(validation_sparse_matrix[validation_index])

        validationSubset = pd.DataFrame()
        validationSubset[class_name] = train_target[validation_index]

        crossValidationPrediction = crossValidationPrediction.reset_index(drop=True)
        validationSubset = validationSubset.reset_index(drop=True)

        crossValidationPrediction[crossValidationPrediction[class_name] < 0.5] = 0
        crossValidationPrediction[crossValidationPrediction[class_name] >= 0.5] = 1

        foldAccuracy = (
                ((crossValidationPrediction[TARGET_LABEL] == validationSubset[TARGET_LABEL]).sum())
                / len(validationSubset))

        print("Prediction based on validation...")
        print("Accuracy for that fold {0}".format(foldAccuracy * 100.0))

        cvAccuracy += foldAccuracy

        # Bagging 5 predictions on the test :)
        if cnt == 1:
            submission[class_name] = model.predict(test_sparse_matrix)
        else:
            submission[class_name] += model.predict(test_sparse_matrix)

cvAccuracy /= 5 # Number of folds = 5 Bagging :)

print("Total CV Accuracy Average of 5 Baggings {0}".format(cvAccuracy * 100.0))

# Checking what is going on

print(submission.head())

submission[TARGET_LABEL] /= 5

submission[ submission.is_multi_author < 0.5] = 0
submission[ submission.is_multi_author >= 0.5] = 1

# Now lets calculate our final accuracy based on PAN Validation file :)

validation_file = pd.read_csv(OFFICIAL_PAN_VALIDATION_FILE)

total_match = (validation_file[TARGET_LABEL] == submission[TARGET_LABEL]).sum()
total_match *= 100.0
total_match /= len(validation_file)

FATALITY_THRESHOLD = 85.00

if total_match >= FATALITY_THRESHOLD:
    print("https://www.youtube.com/watch?v=EAwWPadFsOA&feature=youtu.be&t=15")