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


def lstm(X, y, vocab_size, max_length):

    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_length - 1))
    model.add(CuDNNLSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50, verbose=1)
    return (model)

def sent_generation(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text

    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
    print(in_text)


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

#training the model

#model = lstm(X_train,y_train, vocab_size, max_length)

#saving the model

#model.save('model3_word.h5')
#dump(tokenizer,open('tokenizer3.pkl','wb'))

model = load_model('/storage/home1/e1-246-60/assignment2/model.h5')
tokenizer = load(open('/storage/home1/e1-246-60/assignment2/tokenizer.pkl', 'rb'))


#testing the model

print ("Testing: \n")
#print(test_tokens[:2])
#print('Total Tokens: %d' % len(test_tokens))
#print('Unique Tokens: %d' % len(set(test_tokens)))  


tokenizer_test,X_test,y_test, vocab_size_test, max_length_test = preprocessing(data,test_tokens)
y_test = to_categorical(y_test, num_classes=vocab_size)
#loss = model.evaluate(X_test, y_test, verbose=1)
#print ("loss: ",loss[0])

#perplexity = (2**loss[0])
#print ("perplexity: ",perplexity)

#Sentence generation

temp = randint(0,len(train_tokens)-max_length)
seed_text = train_tokens[temp]

for i in range(1,max_length-1):
    temp = temp+1
    seed_text += ' ' + train_tokens[temp]

print ("seed_text==>   ",seed_text)
n_words = 10
sent_generation(model, tokenizer, max_length-1, seed_text, n_words)






