import json
import os
import glob
import tqdm
import jsonlines
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything

def read_hyperpartisan_data(hyper_file_path):
    """
    Read a jsonl file for Hyperpartisan News Detection data and return lists of documents and labels
    :param hyper_file_path: path to a jsonl file
    :return: lists of documents and labels
    """
    documents = []
    labels = []
    with jsonlines.open(hyper_file_path) as reader:
        for doc in tqdm.tqdm(reader):
            documents.append(doc['text'])
            labels.append(doc['label'])

    return documents, labels

def prepare_hyperpartisan_data(hyper_path='./data/hyperpartisan'):
    """
    Load the Hyperpartisan News Detection data and prepare the datasets
    :param hyper_path: path to the dataset files, {train, dev, test}.jsonl
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(hyper_path):
        raise Exception("Data path not found: {}".format(hyper_path))

    text_set = {}
    label_set = {}
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(hyper_path, split + '.jsonl')
        text_set[split], label_set[split] = read_hyperpartisan_data(file_path)

    enc = preprocessing.LabelBinarizer()
    enc.fit(label_set['train'])
    num_labels = 1 # binary classification
    # vectorize labels as zeros and ones
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.transform(label_set[split])
    return text_set, vectorized_labels, num_labels

def clean_20news_data(text_str):
    """
    Clean up 20NewsGroups text data, from CogLTX: https://github.com/Sleepychord/CogLTX/blob/main/20news/process_20news.py
    // SPDX-License-Identifier: MIT
    :param text_str: text string to clean up
    :return: clean text string
    """
    tmp_doc = []
    for words in text_str.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc

def prepare_20news_data():
    """
    Load the 20NewsGroups datasets and split the original train set into train/dev sets
    :return: dicts of lists of documents and labels and number of labels
    """
    text_set = {}
    label_set = {}
    test_set = fetch_20newsgroups(subset='test', random_state=21)
    text_set['test'] = [clean_20news_data(text) for text in test_set.data]
    label_set['test'] = test_set.target

    train_set = fetch_20newsgroups(subset='train', random_state=21)
    train_text = [clean_20news_data(text) for text in train_set.data]
    train_label = train_set.target

    # take 10% of the train set as the dev set
    text_set['train'], text_set['dev'], label_set['train'], label_set['dev'] = train_test_split(train_text,
                                                                                                train_label,
                                                                                                test_size=0.10,
                                                                                                random_state=21)
    enc = preprocessing.LabelEncoder()
    enc.fit(label_set['train'])
    num_labels = len(enc.classes_)

    # vectorize labels as zeros and ones
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.transform(label_set[split])

    return text_set, vectorized_labels, num_labels

def prepare_eurlex_data(inverted, eur_path='./data/EURLEX57K'):
    """
    Load EURLEX-57K dataset and prepare the datasets
    :param inverted: whether to invert the section order or not
    :param eur_path: path to the EURLEX files
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(eur_path):
        raise Exception("Data path not found: {}".format(eur_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}

    for split in ['train', 'dev', 'test']:
        file_paths = glob.glob(os.path.join(eur_path, split, '*.json'))
        for file_path in tqdm.tqdm(sorted(file_paths)):
            text, tags = read_eurlex_file(file_path, inverted)
            text_set[split].append(text)
            label_set[split].append(tags)

    vectorized_labels, num_labels = vectorize_labels(label_set)

    return text_set, vectorized_labels, num_labels

def read_eurlex_file(eur_file_path, inverted):
    """
    Read each json file and return lists of documents and labels
    :param eur_file_path: path to a json file
    :param inverted: whether to invert the section order or not
    :return: list of documents and labels
    """
    tags = []
    with open(eur_file_path) as file:
        data = json.load(file)
    sections = []
    text = ''
    if inverted:
        sections.extend(data['main_body'])
        sections.append(data['recitals'])
        sections.append(data['header'])

    else:
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])

    text = '\n'.join(sections)

    for concept in data['concepts']:
        tags.append(concept)

    return text, tags

def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None # when genre information is missing

def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                          converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary']) # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test

def prepare_book_summaries(pairs, book_path='data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets
    :param pairs: whether to combine pairs of documents or not
    :param book_path: path to the booksummaries.txt file
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(book_path):
        raise Exception("Data not found: {}".format(book_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    if not pairs:
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]

        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i+1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i+1])))

        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))

        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i+1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i+1])))

    vectorized_labels, num_labels = vectorize_labels(label_set)
    return text_set, vectorized_labels, num_labels

def vectorize_labels(all_labels):
    """
    Combine labels across all data and reformat the labels e.g. [[1, 2], ..., [123, 343, 4] ] --> [[0, 1, 1, ... 0], ...]
    Only used for multi-label classification
    :param all_labels: dict with labels with keys 'train', 'dev', 'test'
    :return: dict of vectorized labels per split and total number of labels
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            all_set.extend(labels)
    all_set = list(set(all_set))

    mlb = MultiLabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)

    print(f'Total number of labels: {num_labels}')

    result = {}
    for split in all_labels:
        result[split] = mlb.transform(all_labels[split])

    return result, num_labels

if __name__ == "__main__":
    seed_everything(3456)
    hyper_text_set, hyper_label_set, hyper_num_labels = prepare_hyperpartisan_data()
    assert hyper_num_labels == 1
    assert len(hyper_text_set['train']) == len(hyper_label_set['train']) == 516
    assert len(hyper_text_set['dev']) == len(hyper_label_set['dev']) == 64
    assert len(hyper_text_set['test']) == len(hyper_label_set['test']) == 65
    news_text_set, news_label_set, news_num_labels = prepare_20news_data()
    assert news_num_labels == 20
    assert len(news_text_set['train']) == len(news_label_set['train']) == 10182
    assert len(news_text_set['dev']) == len(news_label_set['dev']) == 1132
    assert len(news_text_set['test']) == len(news_label_set['test']) == 7532
    eur_text_set, eur_label_set, eur_num_labels = prepare_eurlex_data(False)
    assert eur_num_labels == 4271
    assert len(eur_text_set['train']) == len(eur_label_set['train']) == 45000
    assert len(eur_text_set['dev']) == len(eur_label_set['dev']) == 6000
    assert len(eur_text_set['test']) == len(eur_label_set['test']) == 6000
    inverted_text_set, inverted_label_set, inverted_num_labels = prepare_eurlex_data(True)
    assert inverted_num_labels == 4271
    assert len(inverted_text_set['train']) == len(inverted_label_set['train']) == 45000
    assert len(inverted_text_set['dev']) == len(inverted_label_set['dev']) == 6000
    assert len(inverted_text_set['test']) == len(inverted_label_set['test']) == 6000
    book_text_set, book_label_set, book_num_labels = prepare_book_summaries(False)
    assert book_num_labels == 227
    assert len(book_text_set['train']) == len(book_label_set['train']) == 10230
    assert len(book_text_set['dev']) == len(book_label_set['dev']) == 1279
    assert len(book_text_set['test']) == len(book_label_set['test']) == 1279
    pair_text_set, pair_label_set, pair_num_labels = prepare_book_summaries(True)
    assert pair_num_labels == 227
    assert len(pair_text_set['train']) == len(pair_label_set['train']) == 5115
    assert len(pair_text_set['dev']) == len(pair_label_set['dev']) == 639
    assert len(pair_text_set['test']) == len(pair_label_set['test']) == 639

