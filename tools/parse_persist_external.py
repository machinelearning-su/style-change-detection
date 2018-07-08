import time, argparse, codecs, json, os

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from utils import chunker, print_progress_bar

def main(file_name, output_dir, text_min_chars = 2800, text_max_chars = 5500):
    movies = dict()
    with open(file_name, 'r', errors='ignore') as fdata:
        data_length = sum(1 for line in fdata)

    with open(file_name, 'r', errors='ignore') as fdata:
        tmp_movie_id = None
        movie_id_prefix = 'product/productId: '
        text_prefix = 'review/text: '

        for index, line in enumerate(fdata):
            if(line.startswith(movie_id_prefix)):
                movie_id = line.replace(movie_id_prefix, '', 1)
                tmp_movie_id = movie_id
            elif(line.startswith(text_prefix) and tmp_movie_id):
                text = line.replace(text_prefix, '', 1)
                text = BeautifulSoup(text, 'html.parser').get_text(" ", strip = True)

                if tmp_movie_id not in movies:
                    movies[tmp_movie_id] = [text]
                else:
                    movies[tmp_movie_id].append(text)

                tmp_movie_id = None

            print_progress_bar(index + 1, data_length, description = 'Parsing external data')

    movies_length = len(movies)
    iteration_index = 0

    generated_counter = 0

    for movie_id, texts in movies.items():
        filtered = list(filter(lambda text: len(text) >= text_min_chars and len(text) <= text_max_chars, texts))

        # style change
        for [left_text, right_text, no_change_text] in chunker(filtered, 3):
            if(not no_change_text):
                break

            # with changes - left
            sentences = sent_tokenize(left_text)
            cumulative_length = 0
            curr_sent_index = 0

            left_target_length = int((text_min_chars + text_max_chars) / 4)

            while(cumulative_length < left_target_length):
                cumulative_length += len(sentences[curr_sent_index])
                curr_sent_index += 1

            left_text = ' '.join(sentences[:curr_sent_index])
            change_position = len(left_text)

            # with changes - right
            sentences = sent_tokenize(right_text)
            cumulative_length = 0
            curr_sent_index = 0

            right_target_length = int((text_min_chars + text_max_chars) / 2) - change_position

            while(cumulative_length < right_target_length):
                cumulative_length += len(sentences[curr_sent_index])
                curr_sent_index += 1

            right_text = ' '.join(sentences[:curr_sent_index])

            generated_counter += 1
            name = 'problem-%s' % generated_counter
            persist_entry(' '.join([left_text, right_text]), name, output_dir, [change_position])

            # no changes
            no_change_sentences = sent_tokenize(no_change_text)

            generated_counter += 1
            name = 'problem-%s' % generated_counter
            persist_entry(' '.join(no_change_sentences), name, output_dir, [])

        print_progress_bar(iteration_index + 1, movies_length, description = 'Generating texts')
        iteration_index += 1

    print('Generated data count: %s' % generated_counter)

def persist_entry(text, file_name, output_dir, changes):
    truth = {
        'changes': not len(changes) == 0,
        'positions': changes
    }

    with open('%s%s.txt' % (output_dir, file_name), 'w') as output_txt:
        output_txt.write(text)

    json_truth = json.dumps(truth, indent=8, sort_keys=True)
    with open('%s%s.truth' % (output_dir, file_name), 'w') as output_truth:
        output_truth.write(json_truth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-file', required=True)
    parser.add_argument('-o','--output-dir', required=True)
    args = parser.parse_args()

    if(not args.output_dir.endswith('/')): args.output_dir += '/'

    main(args.input_file, args.output_dir)
