#!/usr/bin/env python
# coding=utf-8

import sys
import json
import xlrd
import os
import jieba
from numpy import *
reload(sys)
sys.setdefaultencoding('utf-8')

class DictBuilder:
    def __init__(self, raw_reviews, seg_reviews):
        self.raw_reviews = raw_reviews
        self.seg_reviews = seg_reviews

    def PreProcess(self):
        with open(self.seg_reviews, 'w') as fw:
            with open(self.raw_reviews) as fr:
                fr = fr.readlines()
                review_num = len(fr)
                review_count = 0
                for review in fr:
                    review = review.strip().decode('utf-8', 'ignore')
                    review_count += 1
                    if len(review) >= 3:
                        cutted_sent = jieba.cut(review)
                        cutted_sent = list(cutted_sent)
                        json_cutted_sent = json.dumps(cutted_sent, ensure_ascii=False)
                        fw.write(json_cutted_sent + '\n')
                        print '{0} finished, total {1}'.format(review_count, review_num)

    def LoadSentiWordStrength(self):
        dut = xlrd.open_workbook(u'../../SentimentAnalysis/Dict/SentimentWords/DUT/file/情感词汇本体.xlsx')
        sheet1 = dut.sheet_by_index(0)
        col1 = sheet1.col_values(0)
        col2 = sheet1.col_values(5)
        words_strength = dict()
        for i, word in enumerate(col1):
            if i == 0:
                continue
            words_strength[col1[i]] = col2[i]
        return words_strength

    def LoadSentiDict(self):
        all_sentwords = set()
        sentiword_polar = dict()
        with open('../../SentimentAnalysis/Dict/SentimentWords/two_motions.txt') as fr:
            for words in fr:
                words = words.strip().split('\t')
                all_sentwords.add(words[0].decode())
                sentiword_polar[words[0].decode()] = int(words[1])
        return all_sentwords, sentiword_polar

    def LoadStopWords(self):
        # 停用词：融合网络停用词、哈工大停用词、川大停用词
        root_path = '../../SentimentAnalysis'
        stop_words = set()
        with open(root_path + u'/Dict/StopWords/file/中文停用词库.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/哈工大停用词表.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/四川大学机器智能实验室停用词库.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/百度停用词列表.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/stopwords_net.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        with open(root_path + u'/Dict/StopWords/file/stopwords_net2.txt') as fr:
            for line in fr:
                item = line.strip().decode()
                stop_words.add(item)
        stop_words.add('')
        stop_words.add(' ')
        stop_words.add(u'\u3000')
        return stop_words

    def ReviewWordExtract(self, seed_num=40):
        stop_words = self.LoadStopWords()
        senti_words, sentiword_polar = self.LoadSentiDict()
        words_strength = self.LoadSentiWordStrength()
        words_freq = dict()
        with open(self.seg_reviews) as fr:
            for line in fr:
                items = json.loads(line.strip())
                for word in set(items):
                    if word not in stop_words:
                        words_freq.setdefault(word, 0)
                        words_freq[word] += 1
        words_freq = sorted(words_freq.iteritems(), key=lambda kk:kk[1], reverse=True)
        # print json.dumps(words_freq[:2000], ensure_ascii=False)
        words, freq = zip(*words_freq)
        review_senti_words = set(words[:2000]) & senti_words
        review_senti_words_freq = dict()
        words_freq = dict(words_freq)
        for word in review_senti_words:
            review_senti_words_freq[word] = {'freq':words_freq[word], 'polar':sentiword_polar[word], 'strength':0 if word not in words_strength else words_strength[word]}
        review_senti_words_freq = sorted(review_senti_words_freq.iteritems(), key=lambda kk:kk[1]['freq'], reverse=True)
        print 'The number of sentiment words from news reviews are {0}'.format(len(review_senti_words_freq))

        # review_senti_words_freq = dict(review_senti_words_freq)
        pos_words_count = neg_words_count = 0
        seed_words = dict()
        for word_info in review_senti_words_freq:
            if pos_words_count < seed_num or neg_words_count < seed_num:
                if word_info[1]['strength'] >= 5:
                    if word_info[1]['polar'] == 1 and pos_words_count < seed_num:
                        pos_words_count += 1
                        seed_words[word_info[0]] = word_info[1]['polar']
                    elif word_info[1]['polar'] == -1 and neg_words_count < seed_num:
                        neg_words_count += 1
                        seed_words[word_info[0]] = word_info[1]['polar']
                    # print word_info[0], word_info[1]

        print 'The number of positive words is {0},\
and the number of negtive words is {1}'.format(pos_words_count, neg_words_count)
        print 'The total number of words is {0}'.format(pos_words_count+neg_words_count)

        seed_words = sorted(seed_words.iteritems(), key=lambda kk:kk[1], reverse=True)

        # print 'The news reviews sentiment words as follows: ', json.dumps(review_senti_words_freq, ensure_ascii=False)
        # print 'The seed sentiment words as follows: ', json.dumps(seed_words, ensure_ascii=False)
        return review_senti_words_freq, seed_words, words_freq

    def UnionWordsFreq(self, word_list):
        with open(self.seg_reviews) as fr:
            fr = fr.readlines()
            review_count = len(fr)
            word_docset = dict()
            for i, line in enumerate(fr):
                line = json.loads(line.strip())
                for word in set(line):
                    word_docset.setdefault(word, set())
                    word_docset[word].add(i)

            bi_word_unionfreq = dict()
            for word1 in word_list:
                for word2 in word_list:
                    # if word1 == word2:
                    # 	continue
                    bi_word_unionfreq[(word1, word2)] = \
                    len(word_docset[word1] & word_docset[word2])

            # print json.dumps(sorted(bi_word_unionfreq.iteritems(), key=lambda kk:kk[1], reverse=True)[:100], ensure_ascii=False)
        return bi_word_unionfreq, review_count




    def SimilarityMatrix(self):
        review_senti_words_freq, seed_words, words_freq = \
        self.ReviewWordExtract(seed_num=40)
        S_words = zip(*seed_words)[0]
        W_words = zip(*review_senti_words_freq)[0]
        bi_word_unionfreq, doc_size = self.UnionWordsFreq(S_words + W_words)

        Y, Y_label = zip(*seed_words)
        Y = mat(Y).T
        Y_label = mat(Y_label).T
        W = len(review_senti_words_freq)
        S = len(seed_words)

        U = zeros((W, S))
        V = zeros((W, W))

        for i, x_word in enumerate(W_words):
            for j, s_y_word in enumerate(S_words):
                U[i][j] = self.PMI(words_freq[x_word], words_freq[s_y_word], \
                bi_word_unionfreq[(x_word, s_y_word)], doc_size)

            for j, w_y_word in enumerate(W_words):
                V[i][j] = self.PMI(words_freq[x_word], words_freq[w_y_word], \
                bi_word_unionfreq[(x_word, w_y_word)], doc_size)

        # 对应分别为：种子情感词向量、新闻评论情感词数（不包括种子情感词）、种子情感词数
        # 新闻评论情感词和种子情感词相似度、新闻评论情感词之间的相似度
        return Y, Y_label, W, S, U, V, S_words, W_words


    def PMI(self, word1_num, word2_num, word1_and_word2_num, doc_size):
        return log2(float((doc_size * word1_and_word2_num +1)) / (word1_num * word2_num))


    def PageRank(self):
        Y, Y_label, W, S, U, V, S_words, W_words = self.SimilarityMatrix()
        P = mat(zeros((W, 1)))
        K = 40
        beta = 0.2
        alpha = 1e-5
        gama = 0.01
        t = 1
        T = 500

        K_max = float('-inf')
        while True:
            for row in V:
                K_max = max(K_max, sorted(row, reverse=True)[K-1])
            V[V<=K_max] = 0.0
            Y_label = Y_label/float((len(Y_label)/2))
            for i, U_row in enumerate(U):
                V_row = V[i]
                U_row_sum = float(sum(map(abs, U_row)))
                V_row_sum = float(sum(map(abs, V_row)))
                # print U_row_sum
                # print V_row_sum
                if U_row_sum == 0:
                    U_row_sum = U_row_sum + 1.0
                if V_row_sum == 0:
                    V_row_sum = V_row_sum + 1.0
                U[i] = U[i]/U_row_sum
                V[i] = V[i]/V_row_sum

            P_pre = P
            P = (1-beta) * U.dot(Y_label) + beta * V.dot(P)
            P = P/float(sum(map(abs, P)))
            delta = sum((array(P - P_pre))**2)

            print 'Iteration:{0}, detal:{1}'.format(t, delta)
            t += 1
            if delta <alpha or t >= T:
                break

        P[P>=0] = 1
        P[P<0] = -1
        return zip(P.tolist(), W_words)

    def LabeledDict(self):
        return self.PageRank()









if __name__=='__main__':
    db = DictBuilder('../data/reviews.txt', '../data/seg_reviews.txt')
    if not os.path.exists('../data/seg_reviews.txt'):
        db.PreProcess()
    db.LabeledDict()
