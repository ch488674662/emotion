#! -*- coding:utf-8 -*-
import re, os, json, codecs, gc
import pandas as pd
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

train_lines = codecs.open('Train_DataSet.csv',encoding='utf-8').readlines()[1:]
print(train_lines)
train_df = pd.DataFrame({
            'id': [x[:32] for x in train_lines],
            'ocr': [x[33:].strip() for x in train_lines]
})
print(train_df)
train_label = pd.read_csv('Train_DataSet_Label.csv')
train_df = pd.merge(train_df, train_label, on='id')

# test_lines = codecs.open('Test_DataSet.csv').readlines()[1:]
# print(test_lines)
# test_df = pd.DataFrame({
#             'id': [x[:32] for x in test_lines],
#             'ocr': [x[33:].strip() for x in test_lines]
# })
