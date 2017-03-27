# -*- encoding:utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import thulac


def cut_words():
    cutter = thulac.thulac(T2S=True, seg_only=True)
    with open('../data/reviews/reviews.txt', 'rb') as fr:
        sys.stdout.write('\r\rStart')
        sys.stdout.flush()
        fw = open('../data/reviews/cut_reviews.txt', 'wb')
        for i, line in enumerate(fr):
            items = line.strip().decode()
            words = cutter.cut(items)
            if len(words) < 2:
                continue
            words, _ = zip(*words)
            fw.write(' '.join(words) + '\n')
            sys.stdout.write('\r\rFinish %d' % i)
            sys.stdout.flush()
        fw.close()


def cut_chars():
    with open('../data/reviews/reviews.txt', 'rb') as fr:
        sys.stdout.write('\r\rStart')
        sys.stdout.flush()
        fw = open('../data/reviews/cut_reviews_char.txt', 'wb')
        for i, line in enumerate(fr):
            items = line.strip().decode()
            words = list(items)
            if len(words) < 2:
                continue
            fw.write(' '.join(words) + '\n')
            sys.stdout.write('\r\rFinish %d' % i)
            sys.stdout.flush()
        fw.close()


if __name__ == '__main__':
    cut_chars()
