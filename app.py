import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import gensim
import pandas as pd

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

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

char_dict = create_dict(codes)

list_of_sentence=[]
for sent in df['Sequence']:
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

def transform(features):
    encode = integer_encoding(features, char_dict) 
    max_length = 100
    pad = tf.keras.preprocessing.sequence.pad_sequences(encode, maxlen=max_length, padding='post', truncating='post')
    return pad


app = Flask(__name__)
model = load_model('convlstm.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.values()

    text = features

    final_features = transform(text)
    prediction = model.predict(final_features)

    

    if prediction>=0.5:
        out = "ACP"
    else:
        out = "Non-ACP"


    return render_template('index.html', prediction_text='The given Peptide should be  {}'.format(out))


if __name__ == "__main__":
    app.run(debug=True)