# -*- encoding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import re
import logging
import gensim
import multiprocessing

def wordsplit(infile, outfile):
	from jieba import cut
	fw = open(outfile, 'w')
	i = 0
	with open(infile) as fr:
		for line in fr:
			res = re.match(r'<content>(.*)</content>', line)
			res_cut = cut(res.group(1))
			fw.write(' '.join(res_cut) + '\n')
			i += 1
			if i%1000 == 0:
				logger.info('Finished {0}'.format(i))
	fw.close()

def train_word2vec_model(infile, ofile1, ofile2):
	from gensim.models import Word2Vec
	from gensim.models.word2vec import LineSentence
	model = Word2Vec(sentences = LineSentence(infile), 
		size=400, window=5, min_count=5, workers=multiprocessing.cpu_count()/2)
	model.save(ofile1)
	model.save_word2vec_format(ofile2, binary=False)


if __name__ == '__main__':
	logging.root.setLevel(logging.INFO)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logger = logging.getLogger(name='news_parse')
	logger.info('Running ' + ' '.join(sys.argv))

	if len(sys.argv) == 3:
		infile, outfile = sys.argv[1:3]
		wordsplit(infile, outfile)

	elif len(sys.argv) == 4:
		infile, ofile1, ofile2 = sys.argv[1:4]
		train_word2vec_model(infile, ofile1, ofile2)

	else:
		logger.error('Wrong args!')