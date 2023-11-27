import os
from transformers import pipeline


# analyzer = pipeline(
#     task='sentiment-analysis',
#     model="cmarkea/distilcamembert-base-sentiment",
#     tokenizer="cmarkea/distilcamembert-base-sentiment"
# )

def sentence_sentiment_label(fname, model_name):
    print('reading file...')
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    sentence_raw = []
    sentences = []
    aspects = []
    aspect_labels = []
    sentence_labels = []
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        text_raw = lines[i].strip()
        aspect = lines[i + 1].lower().strip()
        aspect_label = lines[i + 2].strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        sentence_raw.append(text_raw)
        sentences.append(text)
        aspects.append(aspect)
        aspect_labels.append(aspect_label)
    print('sentiment labeling...')
    analyzer = pipeline("sentiment-analysis", model=model_name)
    results = analyzer(sentences)
    #
    d = {
        'LABEL_0': '-1',
        'LABEL_1': '0',
        'LABEL_2': '1'
    }
    print('reading results...')
    for result in results:
        label = result['label']
        sentence_labels.append(d[label])
    return sentence_raw, sentences, aspects, aspect_labels, sentence_labels


def analysis_consistency(aspect_labels, sentence_labels):
    count = 0
    total_num = len(sentences)
    positive, negative, neural = 0, 0, 0
    for x in range(total_num):
        if int(sentence_labels[x]) == -1:
            negative += 1
        elif int(sentence_labels[x]) == 0:
            neural += 1
        elif int(sentence_labels[x]) == 1:
            positive += 1
    print('postive:', positive)
    print('negative:', negative)
    print('neural:', neural)

    for i in range(len(sentences)):
        if int(sentence_labels[i]) == int(aspect_labels[i]):
            count += 1
    consistency_rate = count / total_num
    return count, consistency_rate


if __name__ == '__main__':
    # model_name='nlptown/bert-base-multilingual-uncased-sentiment'
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    fname = "/home/disk2/jye/ABSA/datasets/implicit/twitter/amazon.txt"
    sentiment_labeled_path = "/home/disk2/jye/ABSA/datasets/implicit/twitter/sentence_sentiment_labeled.txt"
    dirname = os.path.dirname(sentiment_labeled_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    sentence_raw, sentences, aspects, aspect_labels, sentence_labels = sentence_sentiment_label(fname, model_name)
    count, consistency_rate = analysis_consistency(aspect_labels, sentence_labels)
    print(count, consistency_rate)

    print('writing file...')
    with open(sentiment_labeled_path, 'w', encoding='utf-8') as f_1:
        for i in range(len(aspect_labels)):
            f_1.write(sentence_raw[i] + '\n')
            f_1.write(aspects[i] + '\n')
            f_1.write(aspect_labels[i] + '\n')
            f_1.write(sentence_labels[i] + '\n')
        f_1.close()
        print('completed')
