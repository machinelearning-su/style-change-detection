from nltk.tokenize import sent_tokenize
import numpy as np

def get_sentences(text, wordFilter=None):
    sentences = []
    paragraphs = [p for p in text.split('\n') if p]
    for paragraph in paragraphs:
        if wordFilter:
            sentences.extend([wordFilter(s) for s in sent_tokenize(paragraph)])
        else:
            sentences.extend(sent_tokenize(paragraph))

    return sentences

def get_segments(text, n=5, wordFilter=None):
    segments = []
    sentences = get_sentences(text, wordFilter)
    x = len(sentences)
    i = 0
    for i in range(0, x-x%n-n, n):
        segments.append(' '.join(sentences[i:i+n]))
    segments.append(' '.join(sentences[i+n:]))
    
    return segments

def get_sliding_sentences(text, n, wordFilter=None):
    segments = []
    sentences = get_sentences(text, wordFilter)
    x = len(sentences)
    n = min(n, x)
    for i in range(0, x-n+1):
        segments.append(''.join(sentences[i:i+n]))

    return segments

def sent_chunks(X, n=1, wordFilter=None):
    print('Sentence chunks...')
    if n==1:
        return np.array([get_sentences(text, wordFilter) for text in X])
    else:
        return np.array([get_segments(text, n, wordFilter) for text in X])

def sliding_sent_chunks(X, n=5, wordFilter=None):
    print('Sliding sentence chunks...')
    return np.array([get_sliding_sentences(text, n, wordFilter) for text in X])
