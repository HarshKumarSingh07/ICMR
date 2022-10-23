import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import trange
import random
from sklearn import preprocessing
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from keras.models import Model,Sequential
from tensorflow import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.layers import Add, Dense, Input, GlobalMaxPool1D, Conv1D, MaxPooling1D, Embedding,LSTM, Bidirectional, Activation,Dropout, Flatten,BatchNormalization
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
import tensorflow as tf
import gensim
from tensorflow.keras.models import load_model

train = pd.read_excel("Train (1).xlsx")
test = pd.read_excel("Test (1).xlsx")
valid = pd.read_excel("Valid (1).xlsx")

df = pd.concat([train, test, valid])

def create_dict(codes):
      
      char_dict = {}
      for index, val in enumerate(codes):
        char_dict[val] = index+1

      return char_dict

def integer_encoding(data, char_dict):
      
  
  encode_list = []
  for row in data:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list
  
def avg_proba(probas1, probas2, proba3, proba4, proba5):
  final_proba = []
  for i,j,k,l,m in zip(probas1, probas2, proba3, proba4, proba5):
    avg_probas = (i+j+k+l+m)/5
    final_proba.append(avg_probas)
  return final_proba

def my_model(vocabulary_size, embedding_matrix, maximum_length):
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_matrix.shape[1], input_length=maximum_length,weights=[embedding_matrix],trainable=True))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])  
    return model

train = pd.read_excel("Train (1).xlsx")
test = pd.read_excel("Test (1).xlsx")
valid = pd.read_excel("Valid (1).xlsx")

train["Class"] = np.where(train["Class"]=="ACP", 1, 0)
test["Class"] = np.where(test["Class"]=="ACP", 1, 0)
valid["Class"] = np.where(valid["Class"]=="ACP", 1, 0)

train_text = train.Sequence.values
train_labels = train.Class.values

test_text = test.Sequence.values
test_labels = test.Class.values

valid_text = valid.Sequence.values
valid_labels = valid.Class.values


from tensorflow.keras.utils import to_categorical

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))

train_encode = integer_encoding(train_text, char_dict) 
val_encode = integer_encoding(valid_text, char_dict) 
test_encode = integer_encoding(test_text, char_dict) 




max_length = 100
train_pad = tf.keras.preprocessing.sequence.pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
val_pad = tf.keras.preprocessing.sequence.pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
test_pad = tf.keras.preprocessing.sequence.pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')

train_ohe = to_categorical(train_pad)
val_ohe = to_categorical(val_pad)
test_ohe = to_categorical(test_pad)
print(train_pad.shape, val_pad.shape)

list_of_sentence=[]
for sent in train['Sequence']:
    list_new=[]
    for w in sent:
        list_new.append(w)
    list_of_sentence.append(list_new)

Embedding_vector_length = 200
mod = gensim.models.Word2Vec(list_of_sentence, vector_size= Embedding_vector_length, window =5, min_count=1,workers=1, sg = 1)
print(f"Shape of word: {mod.wv.vectors.shape}")
vocabulary_size = len(char_dict) + 1 
embedding_matrix = np.zeros((vocabulary_size, Embedding_vector_length))

for word, i in char_dict.items():
    embedding_matrix[i] = mod.wv[word]

convlstm = my_model(vocabulary_size, embedding_matrix, max_length)

y_train = torch.tensor(train_labels)
y_train = np.array(y_train)
convlstm.fit(train_pad, y_train, epochs = 10)

convlstm.save('convlstm.h5')
print('Model Saved!')
# convlstm = load_model("convlstm.h5")
# preds = convlstm.predict(val_pred)
# print(preds)
# load model
# savedModel=load_model('protcnn.h5')