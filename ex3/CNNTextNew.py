# coding=utf-8
import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.models

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, LSTM, Flatten, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint
# import moxing as mox
import argparse


# BASE_DIR = 'G:\\trainingdata'

parser = argparse.ArgumentParser(description='CNN Example')
# parser.add_argument('--data_url', type=str, default="./Data//",
#                     help='path where the dataset is saved')
# parser.add_argument('--train_url', type=str, default="./Data//", help='if is test, must provide\
#                     path where the trained ckpt file')
parser.add_argument('--data_url', type=str, default=r"E:\Python\NLP\ex3",
                    help='path where the dataset is saved')
parser.add_argument('--train_url', type=str, default=".//", help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args([])  # 会报错，要加上[]
BASE_DIR= args.data_url


#文本语料路径
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')
# 从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
embeddings_index = {}
with open(os.path.join(BASE_DIR, 'glove.6B.100d.txt'),'r',encoding='utf-8') as f:
   for line in f:
       word, coefs = line.split(maxsplit=1)
       coefs = np.fromstring(coefs, 'f', sep=' ')
       embeddings_index[word] = coefs

# 数据预处理
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)
print('Found %s texts.' % len(texts))

# 将样本转化为神经网络训练所用的张量
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
joblib.dump(tokenizer, 'token_result.pkl')

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)
labels = to_categorical(np.asarray(labels))
print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
print(data)
labels = labels[indices]
print(labels)
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
print(data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

# 根据得到的字典生成上文所定义的词向量矩阵
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        # 从预训练模型的词向量到语料库的词向量映射
        embedding_matrix[i] = embedding_vector

# 此处代码不会使用，后面用model.add重新添加形成网络
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)    #输出128维，卷积窗口大小为5

###  补充你的代码开始

model = tensorflow.keras.models.Sequential()
# 加入嵌入层
model.add(Embedding(num_words,
                    EMBEDDING_DIM,
                    weights = [embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = False))
# 添加卷积层
model.add(Conv1D(filters = 128, kernel_size = 5, activation="relu"))
# 添加池化层
model.add(MaxPooling1D(6))
model.add(Conv1D(filters = 128, kernel_size = 5, activation="relu"))
model.add(MaxPooling1D(6))
# 添加Flatten层
model.add(Flatten())
# 添加全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
# 全连接层输出
model.add(Dense(20))
model.add(Activation('softmax'))
# loss使用交叉熵计算，迭代算法用adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# LSTM
# model.add(Embedding(num_words,
#                     EMBEDDING_DIM,
#                     weights = [embedding_matrix],
#                     input_length = MAX_SEQUENCE_LENGTH,
#                     trainable = False))
# # LSTM层
# model.add(LSTM(128,dropout=0.3,recurrent_dropout=0.3))
# # 添加全连接层
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# # 全连接层输出
# model.add(Dense(20))
# model.add(Activation('softmax'))
# # loss使用交叉熵计算，迭代算法用adam
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])

# CNN + LSTM
# 加入嵌入层
# model.add(Embedding(num_words,
#                     EMBEDDING_DIM,
#                     weights = [embedding_matrix],
#                     input_length = MAX_SEQUENCE_LENGTH,
#                     trainable = False))
# # 添加LSTM
# model.add(LSTM(64,return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
# # 添加卷积层
# model.add(Conv1D(filters = 128, kernel_size = 5, activation="relu"))
# # 添加池化层
# model.add(MaxPooling1D(5))
# # 添加Flatten层
# model.add(Flatten())
# # 添加全连接层
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# # 全连接层输出
# model.add(Dense(20))
# model.add(Activation('softmax'))
# # loss使用交叉熵计算，迭代算法用adam
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
###  补充你的代码结束

# checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
history=model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          validation_data=(x_val, y_val))

#fig2=plt.figure()
#plt.plot(history.history['acc'],'r',linewidth=3.0)
#plt.plot(history.history['val_acc'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves : CNN',fontsize=16)
#fig2.savefig('accuracy_cnn.png')
#plt.show()

# Model_DIR=args.train_url
Model_DIR = os.path.join(os.getcwd(), 'mytextcnn_model.h5')
model.save(Model_DIR)
print('Saved model to disk'+Model_DIR)

