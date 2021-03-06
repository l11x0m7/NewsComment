# -*- encoding:utf-8 -*-
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras import models
from keras.layers import LSTM, Dense, GRU, Recurrent
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DLModel():
    def __init__(self, train_path, test_path, embed_size, debug=False, embed_path=None):
        self.embed_size = embed_size
        self.embed_path = embed_path
        self.train_data, self.train_label = self.preprocess(train_path)
        self.test_data, self.test_label = self.preprocess(test_path)
        self.train_sequences, self.test_sequences, \
        self.embed_matrix, self.word_index = \
        self.trans_to_indeces(self.train_data, self.test_data)
        # config
        self.lstm_size =32
        self.label_size = 3
        self.batch_size = 64
        self.epochs = 20
        # debug
        self.debug = debug
        if debug:
            self.lstm_size = 1
            self.epochs = 1


    def preprocess(self, filepath, char_or_word='char'):
        cleaned_data = list()
        cleaned_label = list()
        with open(filepath, 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                label = items[0]
                review = items[1].decode()
                if char_or_word == 'word':
                    import thulac
                    cutter = thulac.thulac(seg_only=True, T2S=True, filt=True)
                    words = cutter.cut(review)
                    if len(words) < 1:
                        continue
                    words, _ = zip(*words)
                elif char_or_word == 'char':
                    words = list(review)
                else:
                    raise ValueError('You must make sure the value of '
                        '[char_or_word] is either char or word')

                if words in ([''], [' ']):
                    continue
                words = map(lambda kk: kk.decode(), words)
                cleaned_data.append(words)
                cleaned_label.append(int(label))

        return cleaned_data, cleaned_label

    def _embeddings_generator(self, path):
        if not path:
            raise ValueError('You must give the path of the word embedding file!')
        with open(path, 'rb') as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                items = line.strip().split()
                word = items[0].decode()
                vector = np.asarray(items[1:], dtype='float32')
                if vector.shape[0] != self.embed_size:
                    raise ValueError(
                            'You must keep the dim of embeds you give same '
                            'as the dim of embeds in embedding file!')
                yield word, vector


    def _to_categories(self, labels):
        array_labels = np.asarray(labels)
        unique_labels = set(labels)
        unique_labels = list(sorted(unique_labels))
        cate_labels = np.zeros([array_labels.shape[0], self.label_size])
        for i, label in enumerate(labels):
            cate_labels[i, unique_labels.index(label)] = 1
        return cate_labels


    def trans_to_indeces(self, train_data, test_data):
        total_data = train_data + test_data
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(total_data)
        word_index = tokenizer.word_index

        embed_matrix = np.zeros([len(word_index) + 1, self.embed_size])
        if self.embed_path:
            for word, embeds in self._embeddings_generator(self.embed_path):
                if word in word_index:
                    embed_matrix[word_index[word]] = embeds


        train_sequences = tokenizer.texts_to_sequences(train_data)
        test_sequences = tokenizer.texts_to_sequences(test_data)
        lengths = [len(sequence) for sequence in train_sequences] + [len(sequence) for sequence in test_sequences]
        max_len = np.max(lengths)
        train_sequences = pad_sequences(train_sequences, maxlen=max_len)
        test_sequences = pad_sequences(test_sequences, maxlen=max_len)

        self.time_step = max_len

        return train_sequences, test_sequences, embed_matrix, word_index

    def model_train(self, train_sequences, train_labels, pre_train=True, ignore_neutral=False):
        from keras.layers import Embedding
        from keras.models import Sequential

        train_labels = np.array(train_labels)
        test_labels = np.array(self.test_label)
        test_sequences = self.test_sequences

        if self.debug:
            train_sequences = train_sequences[:self.batch_size]
            train_labels = train_labels[:self.batch_size]

        if ignore_neutral:
            self.label_size = 2
            train_sequences = train_sequences[train_labels != 0]
            train_labels = train_labels[train_labels != 0]
            test_sequences = test_sequences[test_labels != 0]
            test_labels = test_labels[test_labels != 0]
        else:
            self.label_size = 3
        self.ignore_neutral = ignore_neutral

        if pre_train:
            weights = [self.embed_matrix]
            trainable = False
        else:
            weights = None
            trainable = True
        embedding_layer = Embedding(len(self.word_index) + 1,
                            self.embed_size,
                            weights=weights,
                            input_length=self.time_step,
                            trainable=trainable)

        model = Sequential()
        model.add(embedding_layer)
        # model.add(LSTM(self.lstm_size, return_sequences=True))  # returns a sequence of vectors of dimension 32
        if not self.debug:
            model.add(LSTM(self.lstm_size, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
        model.add(LSTM(self.lstm_size, dropout=0.5, recurrent_dropout=0.5))  # return a single vector of dimension 32
        model.add(Dense(self.label_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        train_labels = self._to_categories(train_labels)
        test_labels = self._to_categories(test_labels)
        model.fit(train_sequences, train_labels,
            batch_size=self.batch_size, epochs=self.epochs,
            validation_data=(test_sequences, test_labels))

        return model

    def model_test(self, test_sequences, test_labels, model):
        test_labels = np.array(test_labels)
        if self.ignore_neutral:
            test_sequences = test_sequences[test_labels != 0]
            test_labels = test_labels[test_labels != 0]

        pred = model.predict(test_sequences)
        test_labels = self._to_categories(test_labels)
        test_labels = np.argmax(test_labels, axis=1)
        pred = np.argmax(pred, axis=1)
        accu = np.mean(pred == test_labels)
        print 'The accuracy of test data is {}'.format(accu)
        print confusion_matrix(test_labels, pred)



def main():
    embed_size = 300
    embed_path = '../wiki/wiki.zh.text.vector.300.char'
    train_path = '../data/reviews/labeled_raw_train.txt'
    test_path = '../data/reviews/labeled_raw_test.txt'
    dlmodel = DLModel(train_path, test_path, embed_size, debug=False, embed_path=embed_path)
    model = dlmodel.model_train(dlmodel.train_sequences, dlmodel.train_label, True, False)
    dlmodel.model_test(dlmodel.test_sequences, dlmodel.test_label, model)
    model = dlmodel.model_train(dlmodel.train_sequences, dlmodel.train_label, True, True)
    dlmodel.model_test(dlmodel.test_sequences, dlmodel.test_label, model)





if __name__ == '__main__':
    main()