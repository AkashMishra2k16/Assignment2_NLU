import numpy
import sys
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import LSTM,CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split

def editing(doc):
    doc = doc.replace('--', ' ')
    #tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    #tokens = [word for word in tokens if word.isalpha()]
    #tokens = [char.lower() for char in tokens]
    return tokens


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
    print("Total Patterns: ", n_patterns)

    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # X = X / float(n_vocab)
    # y = np_utils.to_categorical(dataY)
    y = to_categorical(dataY, num_classes=n_vocab)
    return (X, y, dataX, dataY)


def lstm(X, y):
    model = Sequential()
    model.add(CuDNNLSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [checkpoint]
    model.fit(X, y, epochs=50, batch_size=128)
    return (model)

def sent_generation(model):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
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


filename = "/home1/e1-246-60/assignment2/dataset/austen-emma.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
#print("total_char: ",len(raw_text))
raw_text = raw_text.replace('--', ' ')
table = str.maketrans('', '', string.punctuation)
raw_text = [w.translate(table) for w in raw_text]
print("Finally_total_char: ",len(raw_text))
#print(raw_text[0:2])
#raw_text_train, raw_text_test = train_test_split(raw_text,test_size=0.2)
index = int(0.8*len(raw_text))
raw_text_train = raw_text[0:index]
raw_text_test = raw_text[index:len(raw_text)]

chars = sorted(list(set(raw_text_train)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text_train)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
X,y,dataX,dataY = sampling(n_chars,n_vocab)
#model = Sequential()
#model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
#model.add(Dropout(0.2))
#model.add(Dense(y.shape[1], activation='softmax'))


model = lstm(X,y)
model.save('model_char.h5')
#dump(tokenizer,open('tokenizer.pkl','wb'))

model = load_model('model_char.h5')

sent_generation(model)

chars_test = sorted(list(set(raw_text_test)))
n_chars_test = len(raw_text_test)
n_vocab_test = len(chars_test)
seq_length = 100
dataX_test = []
dataY_test = []

char_to_int_test = dict((c, i) for i, c in enumerate(chars_test))
int_to_char_test = dict((i, c) for i, c in enumerate(chars_test))
for i in range(0, n_chars_test - seq_length, 1):
    seq_in_test = raw_text_test[i:i + seq_length]
    seq_out_test = raw_text_test[i + seq_length]
    dataX_test.append([char_to_int_test[char] for char in seq_in_test])
    dataY_test.append(char_to_int_test[seq_out_test])
n_patterns_test = len(dataX_test)
print ("Total Patterns: ", n_patterns_test)
X_test = numpy.reshape(dataX_test, (n_patterns_test, seq_length, 1))
X_test = X_test / float(n_vocab_test)
#y_test = np_utils.to_categorical(dataY_test)
y_test = to_categorical(dataY_test, num_classes=n_vocab)

model.evaluate(X_test, y_test, batch_size=128, verbose=1)

