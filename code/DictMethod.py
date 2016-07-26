#!/usr/bin/env python
# coding=utf-8
import sys
import os
import time
import xlrd
import json
import numpy as np
import jieba
import MySQLdb
reload(sys)
sys.setdefaultencoding('utf-8')

class DictMethod:
    def __init__(self):
    	self.raw_reviews = '../data/reviews.txt'
    	self.raw_seg_reviews = '../data/raw_seg_reviews.txt'
    	self.conn = MySQLdb.connect(host='localhost', user='root', db='newscomment', charset='utf8')
    	self.cursor = self.conn.cursor()

    def PreProcess(self):
    	with open(self.raw_reviews) as fr:
    		fr = fr.readlines()
    		review_num = len(fr)
    		review_count = 0
    		sql = r"INSERT INTO comments (raw_comment, token_comment) VALUES ('{0}', '{1}');"
    		for review in fr:
				review = review.strip().decode('utf-8', 'ignore')
				if len(review) >= 3:
					cutted_sent = jieba.cut(review)
					cutted_sent = list(cutted_sent)
					json_cutted_sent = '|'.join(cutted_sent).replace(r"'", r'"')
					try:
						self.cursor.execute(sql.format(review, json_cutted_sent))
					except:
						self.conn.commit()
						continue
					review_count += 1
					print '{0} finished, total {1}'.format(review_count, review_num)
		self.conn.commit()

    def LoadSentiDict(self):
		all_sentwords = set()
		sentiword_polar = dict()
		with open('../dict/sentimentwords/two_motions.txt') as fr:
			for words in fr:
				words = words.strip().split('\t')
				all_sentwords.add(words[0].decode())
				sentiword_polar[words[0].decode()] = float(words[1])
		print '[Info] LoadSentiDict Success!'
		return all_sentwords, sentiword_polar

    def LoadStopWords(self):
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
		stop_words.add('')
		stop_words.add(' ')
		stop_words.add(u'\u3000')
		print '[Info] LoadStopWords Success!'
		return stop_words

    def LoadSentiWordStrength(self):
		dut = xlrd.open_workbook(u'../dict/sentimentwords/DUT/file/情感词汇本体.xlsx')
		sheet1 = dut.sheet_by_index(0)
		col1 = sheet1.col_values(0)
		col2 = sheet1.col_values(5)
		words_strength = dict()
		for i, word in enumerate(col1):
			if i == 0:
				continue
			words_strength[col1[i]] = col2[i]
		print '[Info] LoadSentimentWordStrength Success!'
		return words_strength

	# seed_num表示正极性的种子词或者负极性的种子词数，总词数为80个
    def Index(self, seed_num=40, savepath='../dict/comment_emotions.txt'):
		sql = 'SELECT raw_comment, token_comment FROM comments;'
		res = self.cursor.execute(sql)
		word_reviewid = dict()
		words_freq = dict()
		reviews = list()
		if not res:
			print '[Error]: No data fetched from the table "comments"!'

		stop_words = self.LoadStopWords()
		senti_words, sentiword_polar = self.LoadSentiDict()
		words_strength = self.LoadSentiWordStrength()
		ret = self.cursor.fetchall()
		for i, line in enumerate(ret):
			reviews.append(line[0])
			words = line[1].split('|')
			for word in words:
				if word not in stop_words:
					word_reviewid.setdefault(word, dict())
					word_reviewid[word].setdefault(i, 0)
					word_reviewid[word][i] += 1
					words_freq[word] = words_freq.get(word, 0) + 1

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

	print '[Info] The seed words are:', json.dumps(seed_words, ensure_ascii=False)


	pos_seeds = zip(*seed_words)[0][:seed_num/2]
	neg_seeds = zip(*seed_words)[0][seed_num/2:]

	pos_seeds_hits = set()
	neg_seeds_hits = set()
	for word in pos_seeds:
		if word in word_reviewid:
			pos_seeds_hits |= set(word_reviewid[word].keys())
	for word in neg_seeds:
		if word in word_reviewid:
			neg_seeds_hits |= set(word_reviewid[word].keys())

	pos_union_hits = dict()
	neg_union_hits = dict()
	word_score = dict()
	for word in word_reviewid:
		word_union_hit = set()
		if word not in pos_seeds + neg_seeds:
			pos_union_hits[word] = float(len(set(word_reviewid[word].keys())&pos_seeds_hits))
			neg_union_hits[word] = float(len(set(word_reviewid[word].keys())&neg_seeds_hits))
			word_score[word] = round(np.log2((len(neg_seeds_hits)*pos_union_hits[word] + 1)/(len(pos_seeds_hits)*neg_union_hits[word] + 1)), 3)

	word_score = dict(word_score)
	# 归一化
	max_score = max(map(abs, word_score.values()))
	for word in word_score:
		word_score[word] = round(word_score[word] / max_score, 3)

	word_score.update(dict(seed_words))

	word_score = sorted(word_score.iteritems(), key=lambda kk:kk[1], reverse=True)
	print 'The top 100 positive words are as follows: '
	print json.dumps(word_score[:100], ensure_ascii=False)
	print 'The top 100 negative words are as follows: '
	print json.dumps(word_score[-100:], ensure_ascii=False)
	print 'The length of emotion of news comment words is:', len(word_score)
	return word_score

	with open(savepath, 'w') as fw:
		for (word, score) in word_score:
			fw.write(word.encode('utf8') + '\t' + str(score) + '\n')

	def Sentiment(self, comment_word_senti_path='../dict/comment_emotions.txt'):
		word_score = dict()
		if not os.path.exists(comment_word_senti_path):
			word_score = self.Index(seed_num=40, savepath=comment_word_senti_path)
		else:
			with open(comment_word_senti_path) as fr:
				for line in fr:
					items = line.strip().split('\t')
					word_score[items[0].decode()] = float(items[1])

		



if __name__ == '__main__':
    dm = DictMethod()
    # dm.PreProcess()
    dm.Index()
    
