import numpy
import sys
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import to_categorical


def sampling(n_chars, n_vocab):
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text_train[i:i + seq_length]
        seq_out = raw_text_train[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    #print("Total Patterns: ", n_patterns)

    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    #X = X / float(n_vocab)

    y = to_categorical(dataY, num_classes=n_vocab)
    return (X, y, dataX, dataY)


def sent_generation(model):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(100):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        #x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")


filename = "dataset/austen-emma.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
#print("total_char: ",len(raw_text))
raw_text = raw_text.replace('--', ' ')
table = str.maketrans('', '', string.punctuation)
raw_text = [w.translate(table) for w in raw_text]
#print("Finally_total_char: ",len(raw_text))
index = int(0.8*len(raw_text))
raw_text_train = raw_text[0:index]
raw_text_test = raw_text[index:len(raw_text)]

chars = sorted(list(set(raw_text_train)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text_train)
n_vocab = len(chars)
#print ("Total Characters: ", n_chars)
#print ("Total Vocab: ", n_vocab)
X,y,dataX,dataY = sampling(n_chars,n_vocab)


#loading the model
model = load_model('model_char.h5')


#generating sentence
sent_generation(model)




