#!/usr/bin/env python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def washTestData():
    with open('../data/test_reviews.txt') as fr:
        for line in fr:
            items = line.strip().decode()
            if len(items) >= 10:
                print items.encode('utf-8')


if __name__ == '__main__':
    washTestData()

            
