# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import stopwords
import pickle


'''
tokenize and process words in sentences, including lemmatize, lowercase, remove words with non alpha or dight letter
'''
class Lemmatizer(object):
    def __init__(self, basedir=None, output_dir=None):
        self.stop = set(stopwords.words('english'))
        self.wn_lemmatizer = nltk.stem.WordNetLemmatizer()
        self.basedir = basedir
        self.output_dir = output_dir

    def lemmatize(self):
        data_sets = ['train', 'dev', 'test']
        for set_name in data_sets:
            fin_path = self.basedir + 'WikiQA-{}.tsv'.format(set_name)
            fout_path = self.output_dir + 'WikiQA-{}.csv'.format(set_name)
            with open(fin_path, 'r', encoding='utf-8') as fin, open(fout_path, 'w', encoding='utf-8') as fout:
                fin.readline()
                for line in fin:
                    line_info = line.strip().split('\t')
                    q_id = line_info[0]
                    question = line_info[1]
                    a_id = line_info[4]
                    answer = line_info[5]
                    question = ' '.join(self.process_sentence(question))
                    answer = ' '.join(self.process_sentence(answer))
                    label = line_info[6]
                    fout.write('\t'.join([q_id, question, a_id, answer, label]) + '\n')

    def process_sentence(self, line):
        return list(filter(lambda x: x not in self.stop and x.isalnum(),
                        map(lambda x: self.wn_lemmatizer.lemmatize(x).lower(), nltk.word_tokenize(line))))

'''
load qa file from file
'''
def load_qa_data(fname, test=False):
    data = []
    q_dict = {}
    with open(fname, 'r', encoding='utf-8') as fin:
        # this process for test data is for ranking task
        if test:
            for line in fin:
                line_info = line.strip().split('\t')
                q_id = line_info[0]
                question = line_info[1].strip().split()
                answer = line_info[3].strip().split()
                label = int(line_info[4])
                if q_id not in q_dict:
                    q_dict[q_id] = (question, [answer], [label])
                else:
                    q_dict[q_id][1].append(answer)
                    q_dict[q_id][2].append(label)
            for key, value in q_dict.items():
                if 1 in value[2]:
                    data.append(value)
        else:
            for line in fin:
                line_info = line.strip().split('\t')
                question = line_info[1].strip().split()
                answer = line_info[3].strip().split()
                label = int(line_info[4])
                data.append((question, answer, label))
    return data


class DataHelper(object):
    def __init__(self):
        self.vocab = None
        self.word_index = None
        self.idf = None
        self.word_embeddings = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.embedding_dim = None

    def build(self, embedding_file, train_file, dev_file, test_file=None):

        self.train_data = load_qa_data(train_file)
        self.dev_data = load_qa_data(dev_file)
        self.test_data = load_qa_data(test_file, test=True)
        self.build_vocab()
        self.load_embeddings(embedding_file=embedding_file)
        self.transform()

    def build_vocab(self):
        idf = defaultdict(float)
        n = 0
        for question, answer, _ in self.train_data:
            n += 2
            q_words = set(question)
            a_words = set(answer)
            for word in q_words:
                idf[word] += 1
            for word in a_words:
                idf[word] += 1
        for word in idf:
           idf[word] = np.log(n / idf[word])

        self.idf = idf
        self.vocab = list(idf.keys())

        word_index = {}
        for i, word in enumerate(self.vocab):
            word_index[word] = i
        self.word_index = word_index

    # load word embeddings from text file
    def load_embeddings(self, embedding_file):
        word_vecs = {}
        with open(embedding_file, encoding='utf-8') as f:
            i = 0
            for line in f:
                line = line.strip().split()
                word = line[0]
                if word in self.word_index:
                    i += 1
                    embedding = [float(item) for item in line[1:]]
                    word_vecs[word] = np.array(embedding)
        print("num words already in word2vec: " + str(i))
        self.word_embeddings = word_vecs
        self.embedding_dim = len(embedding)

    def word2index(self, sentences):
        if len(sentences) == 0:
            return []
        if isinstance(sentences[0], list):
            tmp = []
            for sentence in sentences:
                tmp.append([self.word_index[word] + 1 for word in sentence if word in self.word_index])
            return tmp
        else:
            return [self.word_index[word] + 1 for word in sentences if word in self.word_index]

    # convert word list to id list
    def transform(self):
        # for sentence padding
        word_embeddings = np.zeros((len(self.vocab) + 1, self.embedding_dim))
        for i, word in enumerate(self.vocab):
            if word in self.word_embeddings:
                word_embeddings[i] = self.word_embeddings[word]
        self.word_embeddings = word_embeddings

        train_tmp = []
        for question, answer, label in self.train_data:
            train_tmp.append((self.word2index(question), self.word2index(answer), label))
        self.train_data = np.array(train_tmp)

        dev_tmp = []
        for question, answer, label in self.dev_data:
            dev_tmp.append((self.word2index(question), self.word2index(answer), label))
        self.dev_data = np.array(dev_tmp)

        test_tmp = []
        for question, answers, labels in self.test_data:
            test_tmp.append((self.word2index(question), self.word2index(answers), labels))
        self.test_data = np.array(test_tmp)

    def save_vocab(self, path):
        pickle.dump(self.vocab, open(path, "wb"))

    def load_vocab(self, path):
        self.vocab = pickle.load( open(path, "rb"))
        word_index = {}
        for i, word in enumerate(self.vocab):
            word_index[word] = i
        self.word_index = word_index

if __name__ == '__main__':
    basedir = './WikiQACorpus/'
    output_dir = './data/'
    lemmatizer = Lemmatizer(basedir, output_dir)
    lemmatizer.lemmatize()