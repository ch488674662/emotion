import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D, Input, Bidirectional,Dropout, Activation, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


train_data = pd.read_csv('Train_DataSet.csv')
train_label = pd.read_csv('Train_DataSet_Label.csv')
train = pd.merge(train_data, train_label, how='left', on='id')
train = train[(train.label.notnull()) & (train.content.notnull())]
test = pd.read_csv('Test_DataSet.csv')

train['title'] = train['title'].fillna('')
train['content'] = train['content'].fillna('')
test['title'] = test['title'].fillna('')
test['content'] = test['content'].fillna('')

import re
def filter(text):
    text = re.sub("[A-Za-z0-9\!\=\？\%\[\]\,\（\）\>\<:&lt;\/#\. -----\_]", "", text)
    text = text.replace('图片', '')
    text = text.replace('\xa0', '') # 删除nbsp
    # new
    r1 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)        #去除html标签
    text = re.sub(r1,'',text)
    text = text.strip()
    return text

def clean_text(data):
    data['title'] = data['title'].apply(lambda x: filter(x))
    data['content'] = data['content'].apply(lambda x: filter(x))
    return data
train = clean_text(train)
test = clean_text(test)

stop_words = pd.read_table('stop.txt', header=None)[0].tolist()

import jieba
import string
table = str.maketrans("","",string.punctuation)
def cut_text(sentence):
    tokens = list(jieba.cut(sentence))
    # 去除停用词
    tokens = [token for token in tokens if token not in stop_words]
#     # 去除英文标点
#     tokens = [w.translate(table) for w in tokens]
    return tokens

train_title = [cut_text(sent) for sent in train.title.values]
train_content = [cut_text(sent) for sent in train.content.values]
test_title = [cut_text(sent) for sent in test.title.values]
test_content = [cut_text(sent) for sent in test.content.values]

all_doc = train_title + train_content + test_title + test_content
print(all_doc)

import gensim
import time
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, save_path):
        self.save_path = save_path
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" %
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
# model_word2vec = gensim.models.Word2Vec.load('final_word2vec_model')
model_word2vec = gensim.models.Word2Vec(min_count=1,
                                        window=5,
                                        size=256,
                                        workers=4,
                                        batch_words=1000)
since = time.time()
model_word2vec.build_vocab(all_doc, progress_per=2000)
time_elapsed = time.time() - since
print('Time to build vocab: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

since = time.time()
model_word2vec.train(all_doc, total_examples=model_word2vec.corpus_count,
                        epochs=5, compute_loss=True, report_delay=60*10,
                        callbacks=[EpochSaver('./final_word2vec_model')])
time_elapsed = time.time() - since
print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_title + test_title)
vocab_size = len(tokenizer.word_index)


error_count=0
embedding_matrix = np.zeros((vocab_size + 1, 256))
for word, i in tqdm(tokenizer.word_index.items()):
    if word in model_word2vec:
        embedding_matrix[i] = model_word2vec.wv[word]
    else:
        error_count += 1


sequence = tokenizer.texts_to_sequences(train_title)
traintitle = pad_sequences(sequence, maxlen=30)
sequence = tokenizer.texts_to_sequences(test_title)
testtitle = pad_sequences(sequence, maxlen=30)
# sequence = tokenizer.texts_to_sequences(train_content)
# traincontent = pad_sequences(sequence, maxlen=512)
# sequence = tokenizer.texts_to_sequences(test_content)
# testcontent = pad_sequences(sequence, maxlen=512)
import tensorflow as tf
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score


#LSTM
import keras
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=256,
                    input_length=30,
                    weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.1)))
model.add(Dense(10))
model.add(Dropout(0.35))
model.add(Dense(3, activation='softmax'))
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=3,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.00001)
model.summary()


#Bilstm
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=256,
                    input_length=30,
                    weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(3, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#TEXTCNN
from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM

from attention import Attention

class TextAttBiRNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims,
                              input_length=self.maxlen, weights=[embedding_matrix])(input)
        x = Bidirectional(LSTM(128, return_sequences=True))(embedding)  # LSTM or GRU
        x = Attention(self.maxlen)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model
model = TextAttBiRNN(maxlen=30, max_features=len(tokenizer.word_index) + 1,
                    embedding_dims=256, class_num=3, last_activation='softmax').get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        # Embedding part can try multichannel as same as origin paper
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen,
                              weights=[embedding_matrix])(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


model = TextCNN(maxlen=30, max_features=len(tokenizer.word_index) + 1,
                embedding_dims=256, class_num=3, last_activation='softmax').get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', metric_F1score])
model.summary()

#传入数据
label = train['label'].astype(int)
# labels = to_categorical(label)
# train_X, val_X, train_Y, val_Y = train_test_split(traintitle, label, shuffle=True, test_size=0.2,
#                                                     random_state=2019)
train_X, val_X, train_Y, val_Y = train_test_split(traintitle, label, shuffle=True, test_size=0.2,
                                                    random_state=2019)
train_Y = to_categorical(train_Y)
model.fit(train_X,
          train_Y,
          batch_size=128,
          epochs=10)

# model.fit(traintitle,
#           labels,
#           batch_size=128,
#           epochs=3,
#           shuffle=True)

from sklearn.metrics import f1_score
pred_val = model.predict(val_X)
print(f1_score(val_Y, np.argmax(pred_val, axis=1), average='macro'))
preds = np.argmax(model.predict(testtitle), axis=1)
test['label'] = preds
test['label'].value_counts()
test[test.label==0]
test[['id', 'label']].to_csv('baseline4.csv', index=False)