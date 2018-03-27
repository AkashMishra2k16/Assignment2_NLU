import numpy as np
from pickle import dump
import string
import math
from random import randint
from pickle import load
from keras.models import load_model
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,CuDNNLSTM
from keras.layers import Embedding
from keras.utils import np_utils
#from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint


def token_gen(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


def preprocessing(data,tokens):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded = tokenizer.texts_to_sequences([data])[0]

    vocab_size = len(tokenizer.word_index) + 1
    #print('Vocabulary Size: %d' % vocab_size)
    # encode 2 words -> 1 word
    in_size = 2
    sequences = list()
    for i in range(in_size, len(encoded)):
        sequence = encoded[i-in_size:i+1]
        sequences.append(sequence)
    #print('Total Sequences: %d' % len(sequences))

    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    #print('Max Sequence Length: %d' % max_length)

    sequences = array(sequences)
    
    X, y = sequences[:,:-1],sequences[:,-1]

    return (tokenizer,X,y, vocab_size, max_length)


file = open("/storage/home1/e1-246-60/assignment2/dataset/austen-emma.txt", 'r')
data = file.read()
file.close()

print ("Training: \n")
tokens = token_gen(data)
#print(len(tokens))
index = int(len(tokens)*0.8)+1
#print ("index: ",index)
train_tokens = tokens[0:index]
test_tokens = tokens[index:len(tokens)]

tokenizer,X_train,y_train, vocab_size, max_length = preprocessing(data,train_tokens)
y_train = to_categorical(y_train, num_classes=vocab_size)


model = load_model('/storage/home1/e1-246-60/assignment2/model.h5')
tokenizer = load(open('/storage/home1/e1-246-60/assignment2/tokenizer.pkl', 'rb'))



#Sentence generation

temp = randint(0,len(train_tokens)-max_length)
seed_text = train_tokens[temp]

for i in range(1,max_length-1):
    temp = temp+1
    seed_text += ' ' + train_tokens[temp]

print ("seed_text==>   ",seed_text)
n_words = 10
sent_generation(model, tokenizer, max_length-1, seed_text, n_words)




