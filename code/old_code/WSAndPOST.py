# -*- encoding=utf-8 -*-
import sys 
import os
import jieba
import json
import urllib2
import time
from argparse import ArgumentParser

"""
	功能：对语料进行分词和词性标注
"""

# 分句：一句话不一定要以句号结尾，由于不关联上下文，所以使用任意标点分句。
def SentenceSegmentation(paragraph):

	cut_list = list(u',.?!:;～，。：；！…？~')
	head = 0
	tail = 0
	sentence_list = list()
	for each in paragraph:
		if each in cut_list:
			if head<tail:
				sentence_list.append(paragraph[head:tail])
			head = tail + 1
			tail = head
		else:
			tail += 1
	if head<tail:
		sentence_list.append(paragraph[head:tail])
	return sentence_list

# 分词：提供结巴分词和ltp分词
def WordSegmentation(sentence, type='jieba'):
	if type == 'jieba':
		from jieba import posseg
		word_tag_list = posseg.cut(sentence.encode('utf-8'))
		word_list = list()
		for each in word_tag_list:
			each = str(each).split('/')
			word_list.append((each[0], each[1]))
		return word_list
	elif type == 'ltp':
		url_get_base="http://api.ltp-cloud.com/analysis/?"
		api_key="Y5a5D3B4xp9ujH4nyDUXvVlNNOCfyuhftwrXWVbA"
		format='plain'
		pattern='ws'	# ws为分词
		result=urllib2.urlopen\
		("%sapi_key=%s&text=%s&format=%s&pattern=%s" % \
			(url_get_base, api_key, sentence.encode('utf-8'), format, pattern))
		content=result.read().strip()
		return content
	else:
		return ""

# 对整个语料分词，并保存
# params:type：分词类型，包括ltp和jieba分词；
# start：起始行；
# AddorWrite：重写文件或末尾追加
def WSofFile(filepath, savepath, type='jieba', start=1, AddorWrite='w'):
	with open(savepath, AddorWrite) as fw:
		with open(filepath) as fr:
			cur = start
			files = fr.readlines()
			while cur <= len(files):
				paragraph = list()
				item = files[cur-1].strip().split('\t')
				sentence_list = SentenceSegmentation(item[0].decode())
				i = 0
				while i < len(sentence_list):
					try:
						word_list = WordSegmentation(sentence_list[i].decode(), type)
					except urllib2.HTTPError as e:
						print e
						print 'Retrying No.%d...' % cur
						continue
					i += 1
					paragraph.append(word_list)
					time.sleep(0.1)
				cur += 1
				print json.dumps(paragraph, ensure_ascii=False)
				fw.write(json.dumps(paragraph, ensure_ascii=False) + '\t' + item[1] + '\n')


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')

	parser = ArgumentParser(description='Word Segment and POS Tagging')
	parser.add_argument('--filepath', dest='filepath')
	parser.add_argument('--savepath', dest='savepath')
	parser.add_argument('--wstype', dest='wstype')
	parser.add_argument('--startline', dest='startline')
	parser.add_argument('--addtype', dest='addtype')
	args = parser.parse_args()
	WSofFile(args.filepath, args.savepath, args.wstype, int(args.startline), args.addtype)


