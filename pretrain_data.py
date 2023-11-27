import os
import sys
import json
import spacy
import yaml
import random
from tqdm import tqdm

sys.path.append(os.pardir)
spacy_en = spacy.load("en_core_web_sm")


def parse_json(data_path, lowercase=False, remove_list=None, max_len=85, min_len=5):
    if remove_list is None:
        remove_list = []
    dataset = []
    d = {
        5: '1',
        3: '0',
        1: '-1'    }
    with open(data_path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            sent = line['text']
            if lowercase:
                sent = sent.lower()
            sentence_len = len(tokenize_en(sent))
            if sentence_len >= max_len:
                continue
            if sentence_len <= min_len:
                continue
            aspect_terms = []
            for term in line['aspect_terms']:
                aspect = term['aspect_term']
                if lowercase:
                    aspect = aspect.lower()
                sentiment = term['sentiment']
                if sentiment in remove_list:
                    continue
                left_index = term['left_index']
                right_index = term['right_index']
                assert aspect == sent[left_index: right_index]
                aspect_terms.append({
                    'aspect_term': aspect,
                    'sentiment': sentiment,
                    'left_index': left_index,
                    'right_index': right_index,
                })
                piece = sent + '__split__' + aspect + '__split__' + d[sentiment] + '__split__' + str(left_index) + \
                        '__split__' + str(right_index)
                dataset.append(piece)
    return dataset


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def analyze(data):
    num = len(data)
    log = {'total': num}
    sentence_lens = []
    aspect_lens = []
    for i in tqdm(range(len(data))):
        text, aspect, sentiment, left_index, right_index = data[i].split('__split__')
        sentence_lens.append(len(tokenize_en(text)))
        aspect_lens.append(len(tokenize_en(aspect)))
        if not sentiment in log:
            log[sentiment] = 0
        log[sentiment] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_min_len'] = min(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log


def save_term_txt(data, save_path):
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    txt = []
    # cnt=0
    for piece in tqdm(data):
        # cnt+=1
        text, term, polarity, start, end = piece.split('__split__')
        start, end = int(start), int(end)
        assert text[start: end] == term
        left_part = text[:start]
        right_part = text[end:]
        txt.append(left_part + '$T$' + right_part)
        txt.append(term)
        txt.append(polarity)

    with open(save_path, 'w') as f:
        for i in txt:
            f.write(i + '\n')
        f.close()


def data_split(full_list, offset, shuffle=True):
    n_total = len(full_list)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist = full_list[:offset]
    return sublist


if __name__ == '__main__':
    base_path = "/home/jye/ABSA/datasets"
    yelp_data_path = os.path.join(base_path, 'yelp/yelp_restaurants.json')
    yelp_save_path = os.path.join(base_path, 'yelp/yelp_restaurants.txt')
    amazon_data_path = os.path.join(base_path, 'amazon/amazon_laptops.json')
    amazon_save_path = os.path.join(base_path, 'amazon/amazon_laptops.txt')

    print('parsing data....')
    yelp_data_new = parse_json(yelp_data_path, remove_list=[2, 4], max_len=85, min_len=5)
    amazon_data_new = parse_json(amazon_data_path, remove_list=[2, 4], max_len=85, min_len=5)
    #
    # print('split data...')
    # yelp_data_new = data_split(yelp_data, offset=150000)
    # amazon_data_new = data_split(amazon_data, offset=150000)

    print('analysing data...')
    log = {
        'yelp': analyze(yelp_data_new),
        'amazon': analyze(amazon_data_new),
        'num_categories': 3,
    }

    print('writing log......')
    with open(os.path.join(base_path, 'pretrain_data_log.log'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)

    print('saving data...')
    save_term_txt(yelp_data_new, yelp_save_path)
    save_term_txt(amazon_data_new, amazon_save_path)
    print('complete')
