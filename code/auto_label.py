# -*- encoding:utf-8 -*-

import sys
import os
import json
import xlrd
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')
import thulac
from collections import defaultdict

# filt : remove the useless words
# T2S : transform 繁体 to 简体
cutter = thulac.thulac(seg_only=True, T2S=True)


# 解析大连理工大学的情感词汇数据
def DUTParser():
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
def HowNetParser():
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
def NTUSDParser():
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

# 哈工大同义词词林解析，用来扩展情感词典
def HITParser():
	hit_list = list()
	with open(u'../dict/sentimentwords/HIT/file/哈工大信息检索研究中心同义词词林扩展版.txt') as fr:
		for line in fr:
			items = line.decode().strip().split(' ')
			hit_list.append(set(items[1:]))
	return hit_list

def PunctuationParser():
	punctuation_dict = defaultdict(int)
	punctuation_dict['?'] = -0.1
	punctuation_dict['？'.decode()] = -0.1
	punctuation_dict['!'] = -0.1
	punctuation_dict['！'.decode()] = -0.1
	return punctuation_dict


def auto_label(filepath, savepath, dut_dict, hownet_dict, ntusd_dict, punctuation_dict):
	labeled_text = list()
	with open(filepath, 'rb') as fr:
		for line in fr:
			line = line.strip()
			cut_line = cutter.cut(line)
			try:
				cut_line, _ = zip(*cut_line)
			except Exception:
				cut_line = [""]
			cut_line = map(lambda kk:kk.decode(), cut_line)

			# DUT
			dut_score = 0.
			# Hownet
			hownet_score = 0.
			# NTUSD
			ntusd_score = 0.
			# punctuation
			punctuation_score = 0.
			for word in cut_line:
				dut_score += dut_dict[word]
				hownet_score += hownet_dict[word]
				ntusd_score += ntusd_dict[word]
				punctuation_score += punctuation_dict[word]

			score_to_label = {True : 1, False : -1}
			dut_label = 0 if dut_score == 0 else score_to_label[dut_score > 0]
			hownet_label = 0 if hownet_score == 0 else score_to_label[hownet_score > 0]
			ntusd_label = 0 if ntusd_score == 0 else score_to_label[ntusd_score > 0]
			punctuation_label = 0 if punctuation_score == 0 else score_to_label[punctuation_score > 0]

			label = dut_label + hownet_label + ntusd_label + punctuation_label

			# print '{0}\t{1}\t{2}\t{3}\t{4}'.format(line, dut_label, hownet_label, ntusd_label, punctuation_label)

			if label != 0:
				label = score_to_label[label > 0]
			labeled_text.append('\t'.join([str(label), line]))

	with open(savepath, 'wb') as fw:
		for labeled_review in labeled_text:
			fw.write(labeled_review + '\n')

def main():
	dut_dict = DUTParser()
	hownet_dict = HowNetParser()
	ntusd_dict = NTUSDParser()
	punctuation_dict = PunctuationParser()
	auto_label('../data/reviews/raw_train.txt', \
			   '../data/reviews/labeled_raw_train.txt',\
			   dut_dict, hownet_dict, ntusd_dict, punctuation_dict)

	auto_label('../data/reviews/raw_test.txt', \
			   '../data/reviews/labeled_raw_test.txt', \
			   dut_dict, hownet_dict, ntusd_dict, punctuation_dict)


if __name__ == '__main__':
	main()
