import os
import sys
import spacy
import yaml
from xml.etree.ElementTree import parse
from lxml import etree
from transformers import BertTokenizer

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
spacy_en = spacy.load("en_core_web_sm")


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def bert_tokenize():
    pretrained_bert_name = 'bert_base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
    return tokenizer

def analyze(data):
    # fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # lines = fin.readlines()
    # fin.close()
    num = len(data)
    sentence_lens = []
    aspect_lens = []
    log = {'total': num}
    for piece in data:
        text, term, polarity, _, _, implicit_label = piece.split('__split__')
        sentence_lens.append(len(tokenize_en(text)))
        aspect_lens.append(len(tokenize_en(term)))
        if not polarity in log:
            log[polarity] = 0
        log[polarity] += 1
        if not implicit_label in log:
            log[implicit_label]=0
        log[implicit_label]+=1
    log['sentence_max_len'] = max(sentence_lens)
    log['sentence_min_len'] = min(sentence_lens)
    log['sentence_avg_len'] = sum(sentence_lens) / len(sentence_lens)
    log['aspect_max_len'] = max(aspect_lens)
    log['aspect_avg_len'] = sum(aspect_lens) / len(aspect_lens)
    return log


def parse_xml(path, lowercase=False, remove_list=None):
    if remove_list is None:
        remove_list = []
    dataset = []
    with open(path, 'rb') as f:
        root = etree.fromstring(f.read())
        for sentence in root:
            index = sentence.get('id')
            sent = sentence.find('text').text
            if lowercase:
                sent = sent.lower()
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            accept_terms = []
            for term in terms:
                aspect = term.attrib['term']
                sentiment = term.attrib['polarity']
                implicit = term.attrib.get('implicit_sentiment', '') == "True"
                if sentiment in remove_list:
                    continue
                left_index = int(term.attrib['from'])
                right_index = int(term.attrib['to'])
                accept_terms.append({
                    'aspect_term': aspect,
                    'sentiment': sentiment,
                    'implicit': implicit,
                    'left_index': left_index,
                    'right_index': right_index,
                })
            if accept_terms:
                dataset.append({
                    'id': index,
                    'text': sent,
                    'aspect_terms': accept_terms,
                })
    return dataset


def parse_sentence_term(path, lowercase=False, implicit=True):
    tree = parse(path)  # 读取XML文件
    sentences = tree.getroot()
    # Element.findall(): 只找到带有标签的元素，该标签是当前元素的直接子元素
    # Element.find() :找到第一个带有特定标签的子元素。
    # Element.text:访问标签的内容
    # Element.get()：访问标签的属性值
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        if lowercase:
            text = text.lower()
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            if lowercase:
                term = term.lower()
            polarity = aspectTerm.get('polarity')
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            implicit_sentiment = aspectTerm.get('implicit_sentiment')
            implicit_sentiment = str(implicit_sentiment)
            if implicit:
                if implicit_sentiment == 'True':
                    piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end + split_char + implicit_sentiment
                    # print(piece)
                    data.append(piece)
            else:
                if implicit_sentiment == 'False':
                    piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end + split_char + implicit_sentiment
                    # print(piece)
                    data.append(piece)
            # piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end + split_char + implicit_sentiment
            # data.append(piece)
    return data


def category_filter(data, remove_list):  # 用来过滤掉conflict类
    remove_set = set(remove_list)
    filtered_data = []
    for text in data:
        if not text.split('__split__')[2] in remove_set:
            filtered_data.append(text)
    return filtered_data


def save_term_txt(data, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence = []
    aspect = []
    label = []
    context = []
    txt = []
    implicit_labels = []
    d = {
        'positive': '1',
        'negative': '-1',
        'neutral': '0',
    }
    d1 = {
        'True': '1',
        'False': '0',
    }
    # cnt=0
    for piece in data:
        # cnt+=1
        text, term, polarity, start, end, implicit_sentiment = piece.split('__split__')
        start, end = int(start), int(end)
        assert text[start: end] == term
        sentence.append(text)
        aspect.append(term)
        label.append(d[polarity])
        implicit_labels.append(d1[implicit_sentiment])
        left_part = text[:start]
        right_part = text[end:]
        context.append(left_part + '$T$' + right_part)
        txt.append(left_part + '$T$' + right_part)
        txt.append(term)
        txt.append(d[polarity])
        txt.append(d1[implicit_sentiment])

    with open(path, 'w') as f:
        for i in txt:
            f.write(i + '\n')
        f.close()


if __name__ == '__main__':
    base_path = "/home/disk2/jye/ABSA/datasets/implicit/laptop/"
    raw_train_path = os.path.join(base_path, 'raw/train.xml')
    raw_test_path = os.path.join(base_path, 'raw/test.xml')
    print('parsing sentence_term......')
    train_data = parse_sentence_term(raw_train_path, lowercase=False, implicit=True)
    test_data = parse_sentence_term(raw_test_path, lowercase=False, implicit=True)

    print('category_filtering......')
    remove_list = ['conflict', 'NULL']
    train_data = category_filter(train_data, remove_list)
    test_data = category_filter(test_data, remove_list)

    print('saving sentence_term_txt......')
    if not os.path.exists(os.path.join(base_path, 'txt2')):
        os.makedirs(os.path.join(base_path, 'txt2'))
    save_term_txt(train_data, os.path.join(base_path, 'txt2/train_implicit_text.txt'))
    save_term_txt(test_data, os.path.join(base_path, 'txt2/test_implicit_text.txt'))

    print('analyzing data......')
    log = {
        'train_data': analyze(train_data),
        'test_data': analyze(test_data),
        'num_categories': 3
    }
    if not os.path.exists(os.path.join(base_path, 'log')):
        os.makedirs(os.path.join(base_path, 'log'))
    print('writing log......')
    with open(os.path.join(base_path, 'log/implicit_log.yml'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)
    print('completed')
