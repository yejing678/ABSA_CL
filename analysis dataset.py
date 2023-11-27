import yaml
import os
import sys
import spacy
from transformers import BertTokenizer

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
spacy_en = spacy.load("en_core_web_sm")


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def bert_tokenize():
    pretrained_bert_name = 'bert_base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    return tokenizer


# BERT tokenizer
# pretrained_bert_name = 'bert_base-uncased'
# tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

def analyze(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    num = len(lines) / 3
    # num = len(lines) / 4
    log = {'total': num}
    sentence_lens = []
    aspect_lens = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        sentence_lens.append(len(tokenize_en(text)))
        aspect_lens.append(len(tokenize_en(aspect)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_min_len'] = min(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log


def analyze_error(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    num = len(lines) / 5
    log = {'total': num}
    sentence_lens = []
    aspect_lens = []
    for i in range(0, len(lines), 5):
        text = lines[i + 1]
        aspect = lines[i + 2].lower().strip()
        ground_truth = lines[i + 3].strip()
        prediction = lines[i + 4]
        sentence_lens.append(len(tokenize_en(text)))
        aspect_lens.append(len(tokenize_en(aspect)))
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_min_len'] = min(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log


if __name__ == '__main__':
    base_path = "/home/disk2/jye/ABSA/Data_analysis/"
    log = {
        'amazon': analyze("/home/disk2/jye/ABSA/datasets/implicit/twitter/amazon.txt"),
        # 'twitter_test': analyze('./datasets/acl-14-short-data/test.raw'),
        'num_categories': 3
    }
    save_path = os.path.join(base_path, 'amazon_analysis_log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('writing log......')
    with open(os.path.join(save_path, 'log.log'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)
    print('completed...')
