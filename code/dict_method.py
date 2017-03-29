# -*- encoding:utf-8 -*-
import sys
import os
reload(sys)
import xlrd
sys.setdefaultencoding('utf-8')

import logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')
import thulac
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

cutter = thulac.thulac(T2S=True, seg_only=True)

# 获取字典内容，包括否定词、程度词、情感词
def stop_words_parser():
    # 停用词：融合网络停用词、哈工大停用词、川大停用词
    stop_words = set()
    with open(u'../dict/stopwords/中文停用词库.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)
    with open(u'../dict/stopwords/哈工大停用词表.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)
    with open(u'../dict/stopwords/四川大学机器智能实验室停用词库.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)
    with open(u'../dict/stopwords/百度停用词列表.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)
    with open(u'../dict/stopwords/stopwords_net.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)
    with open(u'../dict/stopwords/stopwords_net2.txt') as fr:
        for line in fr:
            item = line.strip().decode()
            stop_words.add(item)

    return stop_words

# 解析大连理工大学的情感词汇数据
def dut_parser():
    dut = xlrd.open_workbook(u'../dict/sentimentwords/DUT/file/情感词汇本体.xlsx')
    sheet1 = dut.sheet_by_index(0)
    words = sheet1.col_values(0)
    word_senses_num = sheet1.col_values(2)
    word_emotion_strengths = sheet1.col_values(5)
    word_emotion_types = sheet1.col_values(6)
    dut_dict = defaultdict(int)
    for i, word in enumerate(words):
        if i == 0:
            continue
        word = str(word).decode()
        try:
            word_sense_num = int(word_senses_num[i])
        except:
            word_sense_num = 1
        word_emotion_strength = int(word_emotion_strengths[i]) // 2
        word_emotion_type = int(word_emotion_types[i])
        if word_emotion_type == 1:
            dut_dict[word] += (word_emotion_strength / float(word_sense_num))
        elif word_emotion_type == 2:
            dut_dict[word] -= (word_emotion_strength / float(word_sense_num))

    return dut_dict


# 知网情感词解析
def hownet_parser():
    hownet_dict = defaultdict(int)
    with open(u'../dict/sentimentwords/HowNet/file/正面情感词语（中文）.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            hownet_dict[word] = 1

    with open(u'../dict/sentimentwords/HowNet/file/正面评价词语（中文）.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            hownet_dict[word] = 1

    with open(u'../dict/sentimentwords/HowNet/file/负面情感词语（中文）.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            hownet_dict[word] = -1

    with open(u'../dict/sentimentwords/HowNet/file/负面评价词语（中文）.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            hownet_dict[word] = -1

    return hownet_dict

# 台湾大学情感词典解析
def ntusd_parser():
    ntusd_dict = defaultdict(int)
    with open(u'../dict/sentimentwords/NTUSD/file/ntusd-positive.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            ntusd_dict[word] = 1

    with open(u'../dict/sentimentwords/NTUSD/file/ntusd-negative.txt') as fr:
        for line in fr:
            word = line.strip().decode()
            ntusd_dict[word] = -1

    ntusd_dict[""] = 0

    return ntusd_dict

def dict_parser():
    dut_dict = dut_parser()
    hownet_dict = hownet_parser()
    ntusd_dict = ntusd_parser()
    return dut_dict, hownet_dict, ntusd_dict





class TraditionModel():
    def __init__(self, train_path, test_path):
        self.cleaned_train_data = self.preprocess(train_path)
        self.cleaned_test_data = self.preprocess(test_path)
        self.pos_seed_words, self.neg_seed_words = self.get_seed_words(self.cleaned_train_data)
        self.X_train, self.y_train = self.feature_engineering(self.cleaned_train_data)
        self.X_test, self.y_test = self.feature_engineering(self.cleaned_test_data)


    def preprocess(self, filepath):
        stop_words = stop_words_parser()
        cleaned_data = list()
        with open(filepath, 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                label = items[0]
                review = items[1].decode()
                words = cutter.cut(review)
                if len(words) < 1:
                    continue
                words, _ = zip(*words)
                if words in ([''], [' ']):
                    continue
                words = map(lambda kk: kk.decode(), words)
                cleaned_words = list()
                for word in words:
                    if word not in stop_words:
                        cleaned_words.append(word)
                if len(cleaned_words) < 1:
                    continue
                cleaned_data.append(' '.join(cleaned_words) + '\t' + label)
        return cleaned_data

    def get_seed_words(self, ws_data):
        pos_seed_words = defaultdict(int)
        neg_seed_words = defaultdict(int)
        for sample in ws_data:
            words, label = sample.strip().split('\t')
            label = int(label)
            words = words.split()
            for word in words:
                if label == 1:
                    pos_seed_words[word] += 1
                elif label == -1:
                    neg_seed_words[word] += 1

        pos_seed_words, _ = zip(*filter(lambda kk: kk[1] > 5, pos_seed_words.iteritems()))
        neg_seed_words, _ = zip(*filter(lambda kk: kk[1] > 5, neg_seed_words.iteritems()))

        pos_seed_words = set(pos_seed_words)
        neg_seed_words = set(neg_seed_words)

        return pos_seed_words, neg_seed_words

    def feature_engineering(self, ws_data):
        '''

        正向情感词个数、得分、TF，负向情感词个数、得分、TF，总个数和总得分）
        种子词特征（种子词的选取可以是：在正样本中出现超过5次的词作为positive seed word，
        在负样本中出现超过5次的词作为negative seed word，
        之后统计各个评论里出现的正向种子词频和、负向种子词频和、(正向词数 + 1) / (负向词数 + 1)

        '''

        dicts = dict_parser()

        X = list()
        y = list()
        for sample in ws_data:
            words, label = sample.strip().split('\t')
            label = int(label)
            words = words.split()
            feature = np.zeros([len(dicts), 8])
            seed_feature = np.zeros([3,])
            for word in words:
                if word in self.pos_seed_words:
                    seed_feature[0] += 1
                if word in self.neg_seed_words:
                    seed_feature[1] += 1

                for i in xrange(len(dicts)):
                    if word in dicts[i]:
                        if dicts[i][word] > 0:
                            feature[0][0] += 1
                            feature[0][1] +=dicts[i][word]
                        elif dicts[i][word] < 0:
                            feature[0][3] += 1
                            feature[0][4] +=dicts[i][word]

            # TF和总的词数以及总的得分
            for i in xrange(len(dicts)):
                feature[i][2] = float(feature[i][0]) / len(words)
                feature[i][5] = float(feature[i][3]) / len(words)
                feature[i][6] = feature[i][0] + feature[i][3]
                feature[i][7] = feature[i][1] + feature[i][4]

            seed_feature[2] = (seed_feature[0] + 1) / float(seed_feature[1] + 1)

            feature = feature.flatten()
            total_feature = np.concatenate([feature, seed_feature], axis=0)
            X.append(total_feature)
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        return X, y


    def model_train(self, X_train, y_train, ignore_neutral=False):
        if ignore_neutral:
            X_train = X_train[y_train != 0]
            y_train = y_train[y_train != 0]
        self.ignore_neutral = ignore_neutral

        model = LinearSVC()
        classifier = model.fit(X_train, y_train)
        # pred = classifier.predict(X_train)
        # accu = np.mean(pred == y_train)
        # print 'The accuracy of training data is {}'.format(accu)
        # print confusion_matrix(y_train, pred)

        # k-fold
        kfold = KFold(n_splits=5)
        for i, (train_index, test_index) in enumerate((kfold.split(X_train))):
            X_split_train = X_train[train_index]
            y_split_train = y_train[train_index]
            X_split_valid = X_train[test_index]
            y_split_valid = y_train[test_index]
            classifier = model.fit(X_split_train, y_split_train)
            pred = classifier.predict(X_split_valid)
            accu = np.mean(pred == y_split_valid)
            print 'Fold {} : the accuracy of validation data is {}'.format(i + 1, accu)

        return classifier


    def model_test(self, X_test, y_test, classifier):
        if self.ignore_neutral:
            X_test = X_test[y_test != 0]
            y_test = y_test[y_test != 0]

        pred = classifier.predict(X_test)
        accu = np.mean(pred == y_test)
        print 'The accuracy of test data is {}'.format(accu)

        print confusion_matrix(y_test, pred)


def main():
    train_path = '../data/reviews/labeled_raw_train.txt'
    test_path = '../data/reviews/labeled_raw_test.txt'
    tm = TraditionModel(train_path, test_path)
    print '-' * 20 + '3 kinds of labels' + '-' * 20
    classifier = tm.model_train(tm.X_train, tm.y_train)
    tm.model_test(tm.X_test, tm.y_test, classifier)
    print '-' * 20 + '2 kinds of labels(ignore neutral label)' + '-' * 20
    classifier = tm.model_train(tm.X_train, tm.y_train, True)
    tm.model_test(tm.X_test, tm.y_test, classifier)



if __name__ == '__main__':
    main()












