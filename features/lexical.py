from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from utils import print_progress_bar

def word_len_transformer(data, max_features_len):
    vectors = []

    for entry in data:
        tokens = word_tokenize(entry)
        res = [len(sent) for sent in tokens][:max_features_len]

        res += [0] * (max_features_len - len(res))

        vectors.append(res)

    return vectors

def sent_transformer(data, max_features_len):
    vectors = []

    for entry in data:
        tokens = [word_tokenize(t) for t in sent_tokenize(entry)]

        res = [sum(map(len, sent)) / len(sent) for sent in tokens][:max_features_len]
        res += [0] * (max_features_len - len(res))

        vectors.append(res)

    return vectors

def min_max_lexical_per_sentence(data):
    transformed = []

    data_length = len(data)

    for index, entry in enumerate(data):
        sent_vector = []
        entry_sent = sent_tokenize(entry)

        for sent in entry_sent:
            entry_char = list(sent)
            entry_word = word_tokenize(sent)
            entry_word_tagged = pos_tag(entry_word)

            chars, char_features = lexical_chars(entry_char)
            words, word_features = lexical_words(entry_word_tagged)

            sent_vector.append(chars + words + [entry.count('?'), entry.count('.'), entry.count('!'), len(entry)])

        min_v = np.amin(sent_vector, axis=0).tolist()
        max_v = np.amax(sent_vector, axis=0).tolist()
        transformed.append(np.subtract(max_v, min_v).tolist())

        print_progress_bar(index + 1, data_length, description = 'min_max_lexical_per_sentence')

    return transformed


def lexical(X, feature_names=[]):
    transformed = []

    for i, doc in enumerate(X):
        segments = []

        for entry in doc:
            entry_char = list(entry)
            entry_word = word_tokenize(entry)
            entry_word_tagged = pos_tag(entry_word)
            entry_sent = sent_tokenize(entry)

            chars, char_features = lexical_chars(entry_char)
            words, word_features = lexical_words(entry_word_tagged)
            sentences, sentence_features = lexical_sentences(entry_sent)
            consecutive_dots = [entry.count('..') + entry.count('...') + entry.count('....')]

            segments.append(chars + words + sentences + consecutive_dots)

        transformed.append(segments)

        print_progress_bar(i + 1, len(X), description = 'lexical')

    feature_names.extend(char_features + word_features + sentence_features + ['consecutive_dots'])

    return np.array(transformed)

def lexical_per_sentence(data, max_sent):
    transformed = []

    for entry in data:
        sent_vector = []
        entry_sent = sent_tokenize(entry)

        for sent in entry_sent[:max_sent]:
            entry_char = list(sent)
            entry_word = word_tokenize(sent)
            entry_word_tagged = pos_tag(entry_word)

            sent_vector += lexical_chars(entry_char) + lexical_words(entry_word_tagged)

        sent_vector += [0] * (max_sent * len(sent_vector[0]) - len(sent_vector))
        transformed.append(sent_vector)

    return transformed

def lexical_chars(chars):
    char_count = len(chars)

    possible_chars_map = {
        ',': 'comma_count',
        '\n': 'paragraph_count',
        ';': 'semicolon_count',
        ':': 'colon_count',
        ' ': 'spaces_count',
        '\'': 'apostrophes_count',
        '&': 'amp_count'
    }

    possible_chars = possible_chars_map.keys()

    char_analysis = {
        'digits': 0,
        'punctuation_count': 0,
        'comma_count': 0,
        'semicolon_count': 0,
        'colon_count': 0,
        'spaces_count': 0,
        'apostrophes_count': 0,
        'amp_count': 0,
        'parenthesis_count': 0,
        'paragraph_count': 1
    }

    for char in chars:
        if char in possible_chars:
            char_analysis[possible_chars_map[char]] += 1
        elif char.isdigit(): char_analysis['digits'] += 1
        elif char in ['(', ')']: char_analysis['parenthesis_count'] += 1
        if char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~': char_analysis['punctuation_count'] += 1

    feature_names = list(char_analysis.keys())

    return [char_analysis[key]/char_count for key in feature_names], feature_names

def lexical_words(words_tagged):
    word_count = len(words_tagged)

    word_analysis = {
        'pronouns': 0,
        'prepositions': 0,
        'coordinating_conjunctions': 0,
        'adjectives': 0,
        'adverbs': 0,
        'determiners': 0,
        'interjections': 0,
        'modals': 0,
        'nouns': 0,
        'personal_pronouns': 0,
        'verbs': 0,
        'word_len_gte_six': 0,
        'word_len_two_and_three': 0,
        'avg_word_length': 0,
        'all_caps': 0,
        'capitalized': 0,
        'quotes_count': 0,
    }

    for (word, tag) in words_tagged:
        if tag in ['PRP']: word_analysis['personal_pronouns'] += 1
        if tag.startswith('J'): word_analysis['adjectives'] += 1
        if tag.startswith('N'): word_analysis['nouns'] += 1
        if tag.startswith('V'): word_analysis['verbs'] += 1
        if tag in ['PRP', 'PRP$', 'WP', 'WP$']: word_analysis['pronouns'] += 1
        elif tag in ['IN']: word_analysis['prepositions'] += 1
        elif tag in ['CC']: word_analysis['coordinating_conjunctions'] += 1
        elif tag in ['RB', 'RBR', 'RBS']: word_analysis['adverbs'] += 1
        elif tag in ['DT', 'PDT', 'WDT']: word_analysis['determiners'] += 1
        elif tag in ['UH']: word_analysis['interjections'] += 1
        elif tag in ['MD']: word_analysis['modals'] += 1
        if len(word) >= 6: word_analysis['word_len_gte_six'] += 1
        elif len(word) in [2, 3]: word_analysis['word_len_two_and_three'] += 1
        word_analysis['avg_word_length'] += len(word)
        if word.isupper(): word_analysis['all_caps'] += 1
        elif word[0].isupper(): word_analysis['capitalized'] += 1
        word_analysis['quotes_count'] += word.count('"') + word.count('`') + word.count('\'')

    feature_names = list(word_analysis.keys())

    return [word_analysis[key]/word_count for key in feature_names], feature_names

def lexical_sentences(sentences):
    sent_count = len(sentences)

    sent_analysis = {
        'question_sentences': 0,
        'period_sentences': 0,
        'exclamation_sentences': 0,
        'short_sentences': 0,
        'long_sentences': 0,
        'sentence_length': 0
    }

    for sent in sentences:
        if sent[len(sent) - 1] == '?': sent_analysis['question_sentences'] += 1
        elif sent[len(sent) - 1] == '.': sent_analysis['period_sentences'] += 1
        elif sent[len(sent) - 1] == '!': sent_analysis['exclamation_sentences'] += 1
        if len(sent) <= 100: sent_analysis['short_sentences'] += 1
        elif len(sent) >= 200: sent_analysis['long_sentences'] += 1
        sent_analysis['sentence_length'] += len(sent)

    feature_names = list(sent_analysis.keys())

    return [sent_analysis[key]/sent_count for key in feature_names], feature_names

# CC | Coordinating conjunction |
# CD | Cardinal number |
# DT | Determiner |
# EX | Existential there |
# FW | Foreign word |
# IN | Preposition or subordinating conjunction |
# JJ | Adjective |
# JJR | Adjective, comparative |
# JJS | Adjective, superlative |
# LS | List item marker |
# MD | Modal |
# NN | Noun, singular or mass |
# NNS | Noun, plural |
# NNP | Proper noun, singular |
# NNPS | Proper noun, plural |
# PDT | Predeterminer |
# POS | Possessive ending |
# PRP | Personal pronoun |
# PRP$ | Possessive pronoun |
# RB | Adverb |
# RBR | Adverb, comparative |
# RBS | Adverb, superlative |
# RP | Particle |
# SYM | Symbol |
# TO | to |
# UH | Interjection |
# VB | Verb, base form |
# VBD | Verb, past tense |
# VBG | Verb, gerund or present participle |
# VBN | Verb, past participle |
# VBP | Verb, non-3rd person singular present |
# VBZ | Verb, 3rd person singular present |
# WDT | Wh-determiner |
# WP | Wh-pronoun |
# WP$ | Possessive wh-pronoun |
# WRB | Wh-adverb |
