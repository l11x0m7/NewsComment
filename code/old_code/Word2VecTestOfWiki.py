#-*- encoding:utf-8 -*-
import sys
import os
import time
import logging
import re
from jieba import cut
import gensim
import multiprocessing

reload(sys)
sys.setdefaultencoding('utf-8')

def process_wiki(infile, outfile):
	from gensim.corpora import WikiCorpus
	wiki = WikiCorpus(infile, lemmatize=False, dictionary={})
	i = 0
	with open(outfile, 'w') as fw:
		for text in wiki.get_texts():
			text = ' '.join(text)
			cut_text = cut(text)
			fw.write(re.sub(r' {1,}', ' ', ' '.join(cut_text)) + '\n')
			i += 1
			if i % 1000 == 0:
				logger.info('Saved ' + str(i) + ' texts')
	logger.info('Finished ' + str(i) + ' texts')

def train_word2vec_model(infile, outf1, outf2):
	from gensim.corpora import WikiCorpus
	from gensim.models import Word2Vec
	from gensim.models.word2vec import LineSentence
	model = Word2Vec(sentences = LineSentence(infile), 
		size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
	model.save(outf1)
	model.save_word2vec_format(outf2, binary=False)


if __name__ == '__main__':
	logging.root.setLevel(logging.INFO)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	if len(sys.argv) == 3:
		programname = os.path.basename(sys.argv[0])
		logger = logging.getLogger(name=programname)
		logger.info('Running ' + ' '.join(sys.argv))
		infile, outfile = sys.argv[1:3]
		process_wiki(infile, outfile)
	elif len(sys.argv) == 4:
		programname = os.path.basename(sys.argv[0])
		logger = logging.getLogger(programname)
		logger.info('Running ' + ' '.join(sys.argv))
		infile, outf1, outf2 = sys.argv[1:4]
		train_word2vec_model(infile, outf1, outf2)
	else:
		print 'Error!'
