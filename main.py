#! -*- coding:utf-8 -*-
import re, os, json, codecs, gc
import numpy as np
import pandas as pd
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

train_lines = codecs.open('Train_DataSet.csv',encoding='utf-8').readlines()[1:]
train_df = pd.DataFrame({
            'id': [x[:32] for x in train_lines],
            'ocr': [x[33:].strip() for x in train_lines]
})
train_label = pd.read_csv('Train_DataSet_Label.csv')
train_df = pd.merge(train_df, train_label, on='id')

test_lines = codecs.open('Test_DataSet.csv',encoding='utf-8').readlines()[1:]
test_df = pd.DataFrame({
            'id': [x[:32] for x in test_lines],
            'ocr': [x[33:].strip() for x in test_lines]
})


maxlen = 100
config_path = 'chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/export/home/liuyuzhong/kaggle/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = 'chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

token_dict = {}
#对词典中的词编号
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R

#传入字典
tokenizer = OurTokenizer(token_dict)

#补齐较短的文本,X为一组文本数据
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                #获取文本内容，限定最大长度
                text = d[0][:maxlen]
                # tokenizer.encode('今天天气不错')
                # # 输出是 ([101, 791, 1921, 1921, 3698, 679, 7231, 102], [0, 0, 0, 0, 0, 0, 0, 0])
                x1, x2 = tokenizer.encode(first=text)
                #获取label
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    #迭代器
                    yield [X1, X2], Y[:, 0, :]

                    [X1, X2, Y] = [], [], []

'''tokenizer.tokenize('今天天气不错')
# 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']
tokenizer.encode('今天天气不错')
第一个为wordembedding，第二个为segment embedding
# 输出是 ([101, 791, 1921, 1921, 3698, 679, 7231, 102], [0, 0, 0, 0, 0, 0, 0, 0])'''
from keras.metrics import top_k_categorical_accuracy

#取可能性较大的前k个类别
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        print(l)
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model





def run_cv(nfold, data, data_label, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 3))
    test_model_pred = np.zeros((len(data_test), 3))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]

        model = build_bert(3)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        #一旦学习停止，模型通常会将学习率降低2-10倍，monitor：要监测的测量指标。
        '''
        monitor：要监测的指标量。
        factor：学习速率降低的因素。new_lr = lr * factor
        patience：没有提升的epoch数，之后学习率将降低。
        verbose：int。0：安静，1：更新消息。
        mode：{auto，min，max}之一。在min模式下，当监测量停止下降时，lr将减少；在max模式下，当监测量停止增加时，lr将减少；在auto模式下，从监测数量的名称自动推断方向。
        min_delta：对于测量新的最优化的阀值，仅关注重大变化。
        cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。
        min_lr：学习率的下限。'''
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('bert_dump/' + str(i) + '.hdf5', monitor='val_acc',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=True)

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        # model.load_weights('./bert_dump/' + str(i) + '.hdf5')

        # return model
        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model;
        gc.collect()
        K.clear_session()

        # break

    return train_model_pred, test_model_pred


#one-hot encoding
from keras.utils import to_categorical

DATA_LIST = []
'''iterrows(): 将DataFrame迭代为(insex, Series)对。
itertuples(): 将DataFrame迭代为元祖。
iteritems(): 将DataFrame迭代为(列名, Series)对'''
for data_row in train_df.iloc[:].itertuples():
    # print(data_row.ocr)
    # print(to_categorical(data_row.label, 3))
    DATA_LIST.append((data_row.ocr,to_categorical(data_row.label, 3)))
# print(DATA_LIST)
DATA_LIST = np.array(DATA_LIST)



DATA_LIST_TEST = []
for data_row in test_df.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.ocr, to_categorical(0, 3)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)
train_model_pred, test_model_pred = run_cv(10, DATA_LIST, None, DATA_LIST_TEST)


test_pred = [np.argmax(x) for x in test_model_pred]
test_df['label'] = test_pred
test_df[['id', 'label']].to_csv('baseline3.csv', index=None)