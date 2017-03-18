# -*- encoding:utf-8 -*-
import os
import sys
import numpy as np
import json
import logging

filepath = '../data/reviews/reviews.txt'
choosed_indeces = '../data/reviews/choosed_indeces.npz'

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')

def _load_data(filepath):
    reviews = list()
    with open(filepath, 'rb') as fr:
        for line in fr:
            review = line.strip()
            reviews.append(review)
        return reviews


def _sample_data(savepath, reviews, sample_number=0):
    if sample_number == 0:
        return
    with open(savepath, 'ab') as fw:
        if os.path.exists(choosed_indeces):
            choosed = json.load(open(choosed_indeces, 'rb'))
        else:
            choosed = list()

        assert len(choosed) == len(set(choosed))

        total = range(len(reviews))

        left = list(set(total) - set(choosed))


        random_indeces = np.random.choice(left, sample_number, replace=False)
        random_sample = np.array(reviews)[random_indeces]

        choosed.extend(random_indeces)

        json.dump(choosed, open(choosed_indeces, 'wb'))

        fw.write('\n'.join(map(lambda kk:'\t' + kk, random_sample)) + '\n')


def _sample_train_data(trainpath, reviews, sample_number):
    _sample_data(trainpath, reviews, sample_number)
    logging.log(logging.INFO, 'train done')


def _sample_test_data(testpath, reviews, sample_number):
    _sample_data(testpath, reviews, sample_number)
    logging.log(logging.INFO, 'test done')




def sample_train_test_data(trainpath, testpath, train_number, test_number):
    reviews = _load_data(filepath)
    _sample_train_data(trainpath, reviews, train_number)
    _sample_test_data(testpath, reviews, test_number)


def main():
    train_path = '../data/reviews/raw_train.txt'
    test_path = '../data/reviews/raw_test.txt'
    train_number = 2200
    test_number = 800
    sample_train_test_data(train_path, test_path, train_number, test_number)


if __name__ == '__main__':
    main()