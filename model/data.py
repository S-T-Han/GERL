from sklearn import neighbors
from sympy import Order
import torch
import dgl

import numpy as np
import os
import time
import nltk
import json


class MIND():
    def __init__(self):
        self.data_path = os.getcwd() + '/data/'
        self.behaviors_path = {'train': 'train/behaviors.tsv', 'dev': 'dev/behaviors.tsv'}
        self.news_path = {'train': 'train/news.tsv', 'dev': 'dev/news.tsv'}
        self.user_id_vocab_path = 'dgl/vocab/user_id_vocab.json'
        self.news_id_vocab_path = 'dgl/vocab/news_vocab.json'
        self.topic_vocab_path = 'dgl/vocab/topic_vocab.json'
        self.word_vocab_path = 'dgl/vocab/word_vocab.json'
        self.graphs_path = {'train': 'dgl/train/graph.bin', 'dev': 'dgl/dev/graph.bin'}

        if os.path.exists(self.data_path + 'dgl'):
            print('Find preprocessed files')
            with open(self.data_path + self.user_id_vocab_path, 'r', encoding='utf-8') as file:
                self.user_id_vocab = json.load(file)
            with open(self.data_path + self.news_id_vocab_path, 'r', encoding='utf-8') as file:
                self.topic_vocab = json.load(file)
            with open(self.data_path + self.topic_vocab_path, 'r', encoding='utf-8') as file:
                self.topic_vocab = json.load(file)
            with open(self.data_path + self.word_vocab_path, 'r', encoding='utf-8') as file:
                self.word_vocab = json.load(file)
            self.graphs = {'train': self.loadGraph('train'), 'dev': self.loadGraph('dev')}
        else:
            print('Preprocessed files do not exist')
            self.behaviors = {
                'train': self.preprocessBehaviors('train'), 'dev': self.preprocessBehaviors('dev')}
            self.news = {'train': self.preprocessNews('train'), 'dev': self.preprocessNews('dev')}
            self.user_id_vocab = self.buildUserIdVocab()
            self.news_id_vocab, self.topic_vocab, self.word_vocab = \
                self.buildNewsVocabs()
            self.graphs = {'train': self.buildGraph('train'), 'dev': self.buildGraph('dev')}
            self.save()

    def buildGraph(self, subset):
        print('Building {} graph...'.format(subset))
        behaviors, news = self.behaviors[subset], self.news[subset]
        src, dst = [], []
        user_id_idx = []
        news_id_idx, topic_idx, title_idx = [], [], []
        news_id_to_news_nid, nid = {}, 0
        for user_nid, (user_id, imp_log) in enumerate(behaviors.items()):
            user_id_idx.append(self.user_id_vocab.get(user_id, 1))
            for news_id in imp_log:
                topic, title = news[news_id]
                if news_id not in news_id_to_news_nid.keys():
                    news_id_to_news_nid.update({news_id: nid})
                    news_id_idx.append(self.news_id_vocab.get(news_id, 1))
                    topic_idx.append(self.topic_vocab.get(topic, 1))
                    title_idx.append(
                        [(self.word_vocab.get(title[i], [1, np.inf])[0] if i < len(title) else 0) 
                        for i in range(0, 30)])
                    nid += 1
                news_nid = news_id_to_news_nid[news_id]
                src.append(user_nid)
                dst.append(news_nid)

        graph = dgl.heterograph({('user', 'clicked', 'news'): (torch.tensor(src), torch.tensor(dst))})
        graph.ndata['user_id'] = {'user': torch.tensor(user_id_idx)}
        graph.ndata['news_id'] = {'news': torch.tensor(news_id_idx)}
        graph.ndata['topic'] = {'news': torch.tensor(topic_idx)}
        graph.ndata['title'] = {'news': torch.tensor(title_idx)}

        return graph

    def buildUserIdVocab(self):
        print('Building user_id vocab...')
        user_id_vocab = {key: (i + 2) for i, key in enumerate(self.behaviors['train'].keys())}
        user_id_vocab.update({'<pad>': 0, '<unk>': 1})

        return user_id_vocab

    def buildNewsVocabs(self):
        print('Building news vocab')
        news_id_vocab, num_news_id = {'<pad>': 0, '<unk>': 1}, 2
        topic_vocab, num_topics = {'<pad>': 0, '<unk>': 1}, 2
        word_vocab, num_words = {'<pad>': [0, np.inf], '<unk>': [1, np.inf]}, 2
        for news_id, value in self.news['train'].items():
            topic, title = value
            news_id_vocab.update({news_id: num_news_id})
            num_news_id += 1
            if topic not in topic_vocab.keys():
                topic_vocab.update({topic: num_topics})
                num_topics += 1
            for word in title:
                if word in word_vocab.keys():
                    word_vocab[word][1] += 1
                else:
                    word_vocab.update({word: [num_words, 1]})
                    num_words += 1

        return news_id_vocab, topic_vocab, word_vocab

    def preprocessBehaviors(self, subset):
        print('Preprocessing {} behaviors...'.format(subset))
        file_path = self.data_path + self.behaviors_path[subset]
        behaviors = {}
        with open(file_path, encoding='utf-8') as file:
            for i, line in enumerate(file):
                line = line.strip('\n').split('\t')
                user_id, imp_log = line[1], line[4].split(' ')
                imp_log = [log.split('-')[0] for log in imp_log if log.split('-')[1] == '1']
                behaviors.update({user_id: imp_log})

        return behaviors

    def preprocessNews(self, subset):
        print('Preprocessing {} news...'.format(subset))
        file_path = self.data_path + self.news_path[subset]
        news = {}
        with open(file_path, encoding='utf-8') as file:
            for i, line in enumerate(file):
                line = line.strip('\n').split('\t')
                news_id, topic, title = line[0], line[1], nltk.word_tokenize(line[3].lower())
                news.update({news_id: (topic, title)})

        return news

    def save(self):
        print('Saving everything...')
        os.makedirs(self.data_path + 'dgl/vocab')
        for subset in ('train', 'dev'):
            os.makedirs(self.data_path + 'dgl/' + subset)
        with open(self.data_path + self.user_id_vocab_path, 'w', encoding='utf-8') as file:
            json_str = json.dumps(self.user_id_vocab, indent=4)
            file.write(json_str)
        with open(self.data_path + self.news_id_vocab_path, 'w', encoding='utf-8') as file:
            json_str = json.dumps(self.topic_vocab, indent=4)
            file.write(json_str)
        with open(self.data_path + self.topic_vocab_path, 'w', encoding='utf-8') as file:
            json_str = json.dumps(self.topic_vocab, indent=4)
            file.write(json_str)
        with open(self.data_path + self.word_vocab_path, 'w', encoding='utf-8') as file:
            json_str = json.dumps(self.word_vocab, indent=4)
            file.write(json_str)
        for subset in ('train', 'dev'):
            dgl.save_graphs(self.data_path + self.graphs_path[subset], [self.graphs[subset]])

    def loadGraph(self, subset):
        glist, _ = dgl.load_graphs(self.data_path + self.graphs_path[subset])

        return glist[0]


class GolVe():
    def __init__(self):
        self.vocab_path = os.getcwd() + '/.vector_cache/glove.6B.300d.txt'
        self.embedding = self.build_embedding()

    def build_embedding(self):
        vocab = {}
        with open(self.vocab_path, encoding='utf-8') as file:
            for i, line in enumerate(file):
                line = line.strip('\n').split(' ')
                word, emb = line[0], torch.tensor([float(num) for num in line[1:]], dtype=torch.float32)
                vocab.update({word: emb})

        return vocab


if __name__ == '__main__':
    mind = MIND()