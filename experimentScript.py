import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout, RNN, SimpleRNN
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Layer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend
from sklearn import preprocessing
from sklearn.metrics import log_loss
import gc
import matplotlib.pyplot as plt
#%matplotlib inline
import os


train_df = pd.read_csv("/home/s1779494/MLP/data/train.csv")
test_df = pd.read_csv("/home/s1779494/MLP/data/test.csv")


def train_model_and_plot_stats(history, nameOfFile=" ", model=None):
    print('got here')
    lossFileName = nameOfFile + ' loss.pdf';
    accuracyFileName = nameOfFile + ' accuracy.pdf';
    val_accuracy_on_validation = nameOfFile + ' accuracy on validation set';
    file_model_sumary = nameOfFile + ' model sumary';

    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(accuracyFileName)
    plt.show()


    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(lossFileName)
    plt.show()

    model_summary = str(model.to_json())
    print(type(model_summary))
    print(type(model.summary()))
    np.savetxt(val_accuracy_on_validation, (history.history['val_acc']), fmt='%.18e', delimiter=' ', newline=os.linesep)
    # np.savetxt(file_model_sumary , (model_summary), fmt='%.18e',delimiter=' ', newline=os.linesep)
    text_file = open(file_model_sumary, "w")
    text_file.write(model_summary)
    text_file.close()
    return


def get_lstm_feats(a=20000, b=10, c=300, bat=32, seed = 42, run = 1):
    # return train pred prob and test pred prob
    NUM_WORDS = a
    N = b
    MAX_LEN = c
    NUM_CLASSES = 3
    nameOfFile = 'SimpleRNN 1 layer early stopping Adam LR 0_0001 opt Run ' + str(run)

    X = train_df['text']
    Y = train_df['author']
    X_test = test_df['text']

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(X)

    train_x = tokenizer.texts_to_sequences(X)
    train_x = pad_sequences(train_x, maxlen=MAX_LEN)

    test_x = tokenizer.texts_to_sequences(X_test)
    test_x = pad_sequences(test_x, maxlen=MAX_LEN)

    lb = preprocessing.LabelBinarizer()
    lb.fit(Y)

    train_y = lb.transform(Y)

    model = Sequential()
    model.add(Embedding(NUM_WORDS, N, input_length=MAX_LEN))
    #     model.add(LSTM(N, dropout=0.2, recurrent_dropout=0.2))
    model.add(SimpleRNN(N, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                        recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False,
                        stateful=False, unroll=False))
    # model.add(Dense(N)) # this is a fully-connected layer with N hidden units.
    #  model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    optim = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.summary()  # prints a summary representation of your model.

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    modelName = "[model] " + nameOfFile
    model_chk = ModelCheckpoint(filepath=modelName, monitor='val_loss', save_best_only=True,
                                verbose=1)  # Save the model after every epoch.
    np.random.seed(seed)
    history = model.fit(train_x, train_y,
                        validation_split=0.1,
                        batch_size=bat, epochs=500,
                        verbose=2,
                        callbacks=[model_chk, earlyStopping],
                        shuffle=True
                        )

    print(history.history['val_acc'])
    train_model_and_plot_stats(history, nameOfFile, model=model)

    #     model = load_model(modelName)
    #     train_pred = model.predict(train_x)
    #     test_pred = model.predict(test_x)
    del model
    gc.collect()
#     print(log_loss(train_y,train_pred))
#     return train_pred,test_pred


for i in range(5):
    backend.clear_session()
    get_lstm_feats(16000,12,300,256,seed=42*i,run=i)