import numpy as np
#import pickle
from pickle import dump
import string
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
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded = tokenizer.texts_to_sequences([data])[0]
    #print((tokenizer.word_docs))
    # retrieve vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # encode 2 words -> 1 word
    in_size = 2
    sequences = list()
    for i in range(in_size, len(encoded)):
        sequence = encoded[i-in_size:i+1]
        sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))
    # pad sequences
    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)
    # split into input and output elements
    sequences = array(sequences)
    print(sequences[0])
    print(sequences[0][0])
    X, y = sequences[:,:-1],sequences[:,-1]
    #print(y.shape)
    #y = to_categorical(y, num_classes=vocab_size)
    #y = np_utils.to_categorical(y)
    return (tokenizer,X,y, vocab_size, max_length)


def lstm(X, y, vocab_size, max_length):
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_length - 1))
    model.add(CuDNNLSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))
    # print(model.summary())
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # filepath="word_weights_improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    # fit network
    model.fit(X, y, epochs=50, verbose=1)
    return (model)

def sent_generation(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    #result = list()
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        #result.append(out_word)
    print(in_text)
    #return (' '.join(result))


file = open("/home1/e1-246-60/assignment2/dataset/austen-emma-persuasion-sense.txt", 'r')
data = file.read()
file.close()
#print(len(data))

#data_train,data_test = train_test_split(data,test_size=0.2)

#train_tokens = token_gen(data_train)
tokens = token_gen(data)
print(len(tokens))
#print(tokens[0:3])
#train_tokens, test_tokens = train_test_split(tokens, test_size= 0.2)
train_tokens = tokens[0:290603]
test_tokens = tokens[290603:len(tokens)]
#print(train_tokens[0:3])
#print('Total Tokens: %d' % len(train_tokens))
#print('Unique Tokens: %d' % len(set(train_tokens)))

#sequences = seq_gen(train_tokens)

tokenizer,X_train,y_train, vocab_size, max_length = preprocessing(data,train_tokens)
y_train = to_categorical(y_train, num_classes=vocab_size)


#Training the model
model = lstm(X_train,y_train, vocab_size, max_length)
#saved(model,tokenizer)
model.save('model3_word.h5')
dump(tokenizer,open('tokenizer3.pkl','wb'))


#dump(tokenizer,open('tokenizer.pkl','wb'))
model = load_model('model3_word.h5')
tokenizer = load(open('tokenizer3.pkl', 'rb'))


##testing the model
#test_tokens = token_gen(data_test)
print(test_tokens[:2])
print('Total Tokens: %d' % len(test_tokens))
print('Unique Tokens: %d' % len(set(test_tokens)))

#sequences = seq_gen(test_tokens)

tokenizer_test,X_test,y_test, vocab_size_test, max_length_test = preprocessing(data,test_tokens)
y_test = to_categorical(y_test, num_classes=vocab_size)
model.evaluate(X_test, y_test, verbose=1)


temp = randint(0,len(train_tokens)-max_length)
seed_text = train_tokens[temp]

for i in range(1,max_length-1):
    temp = temp+1
    seed_text += ' ' + train_tokens[temp]

#print(seed_text)
n_words = 10
sent_generation(model, tokenizer, max_length-1, seed_text, n_words)
#print(sent)
