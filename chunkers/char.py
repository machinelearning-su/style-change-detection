import numpy as np

def split_seq(seq, depth, breath=None):
    if not breath:
        breath = round(len(seq) / depth)
    newseq = []
    splitsize = (1.0/breath)*len(seq)
    for i in range(breath):
        newseq.append(' '.join(seq[round(i*splitsize):round((i+1)*splitsize)]))
    return newseq

def get_segments(text, n, chunks):
    segments = split_seq(text, n, chunks)
    return segments

def get_sliding_chars(text, n=None, chunks=None, wordFilter=None, process=False):
    segments = []
    mult = 3

    if n:
        n = min(n, len(text))
        chunks = round(len(text) / n)
    
    parts = chunks * mult
    part_size = round(len(text) / parts)
    i = 0
    for i in range(0, parts - mult):
        segments.append(''.join(text[i*part_size:i*part_size+mult*part_size]))
    segments.append(''.join(text[i*part_size:]))
    
    return segments

def char_chunks(X, n=None, chunks=None, sliding=False):
    chunker = get_sliding_chars if sliding else get_segments
    
    return np.array([chunker(text, n, chunks) for text in X])
