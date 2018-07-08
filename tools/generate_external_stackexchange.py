import pandas as pd
import numpy as np
import random

from bs4 import BeautifulSoup

from os import listdir
from os.path import isfile, join
import re

def parse_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    # Replace URLs with actual URL
    for link in soup.find_all('a', href = True):
        link.replace_with(link["href"])
    # Remove code, quotes, JS, styles
    for tag in soup.find_all(['code', 'blockquote', 'script', 'style']):
        tag.decompose()
    res = soup.get_text()
    return res

CSV_PATH = "F:\\Stack Exchange Data"

THRESHOLD = 2000

def generate_examples(frame):
    questions = frame.groupby('Id')
    examples = []
    for question in questions:
        posts = question[1]['Body']
        long_posts = []
        short_posts = []
        for post in posts:
            if len(post) > THRESHOLD:
                long_posts.append(post)
            else:
                short_posts.append(post)
        if len(long_posts) == 0:
            continue
        long_posts = sorted(long_posts, key=len)
        long_posts.reverse()
        short_posts = sorted(short_posts, key=len)
        short_posts.reverse()

        count_long = len(long_posts)
        count_short = len(short_posts)
        if 2 * count_long <= count_short:
            for i in range(count_long):
                examples.append([long_posts[i], short_posts[i], short_posts[i + count_long]])
        else:
            num_triples = (count_long + count_short) // 3
            all_posts = long_posts + short_posts
            for i in range(num_triples):
                candidate = [all_posts[i], all_posts[i + num_triples], all_posts[i + 2 * num_triples]]
                if len(candidate[0]) > THRESHOLD:
                    examples.append(candidate)
    print(len(examples))
    return examples


def main():
    files = [f for f in listdir(CSV_PATH) if isfile(join(CSV_PATH, f))]
    files = [f for f in files if re.match("Query_.*csv", f)]
    print("Read %d files" % len(files))

    all_examples = []
    for file in files:
        print(file)
        frame = pd.read_csv(join(CSV_PATH, file))
        # Strip html
        frame['Body'] = frame['Body'].apply(lambda text: parse_html(text))
        examples = generate_examples(frame)
        all_examples.extend(examples)

    total_len = len(all_examples)
    print("Prepared %d examples" % total_len)

    fraction = total_len // 7
    choices = []
    choices.extend([2] * (fraction * 2))
    choices.extend([1] * fraction)
    choices.extend([0] * (total_len - fraction * 3))
    random.shuffle(choices)

    df = pd.DataFrame(index=np.arange(0, total_len),
                      columns=['text', 'positions'])
    curr = 0
    for example in all_examples:
        if choices[curr] == 0:
            splits = []
            text = example[0]
        elif choices[curr] == 1:
            splits = [len(example[0])]
            text = example[0] + example[1]
        else:
            splits = [len(example[0]), len(example[0]) + len(example[1])]
            text = "".join(example)
        df.loc[curr] = {'text': text, 'positions': splits}
        curr = curr + 1
    df.head()
    df.to_csv(join(CSV_PATH, "result.csv"))

if __name__ == "__main__":
    main()

