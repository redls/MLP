import numpy as np
import pandas as pd
from keras import optimizers
from keras import regularizers
from keras.layers import Embedding, LSTM, Dense, Flatten, Dropout, RNN, SimpleRNN, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Layer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend
from sklearn import preprocessing
from sklearn.metrics import log_loss
import gc
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt



train_df = pd.read_csv("/home/s1779494/MLP/experiments/train.csv")
test_df = pd.read_csv("/home/s1779494/MLP/experiments/test.csv")


def train_model_and_plot_stats(history, nameOfFile=" ", model=None):
    lossFileName = '/home/s1779494/MLP/experiments/results/' +  nameOfFile + ' loss.pdf';
    accuracyFileName = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' accuracy.pdf';
    val_accuracy_on_validation = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' accuracy on validation set';
    file_model_sumary = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' model sumary';
    val_accuracy_on_training = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' accuracy on training set'
    val_loss_on_validation = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' loss on validation set'
    val_loss_on_training = '/home/s1779494/MLP/experiments/results/' + nameOfFile + ' loss on training set'
    #  "Accuracy"
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    ax_1.plot(history.history['acc'])
    ax_1.plot(history.history['val_acc'])
    ax_1.legend(['Train', 'Validation'], loc=0)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('Accuracy')
    
    # "Loss"
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    ax_2.plot(history.history['loss'])
    ax_2.plot(history.history['val_loss'])
    ax_2.legend(['Train', 'Validation'], loc=0)
    ax_2.set_ylabel('Loss')
    ax_2.set_xlabel('Epoch number')
    
    fig_1.tight_layout() 
    fig_2.tight_layout() 

    fig_1.savefig(accuracyFileName) 
    fig_2.savefig(lossFileName) 


    model_summary = str(model.to_json())
    print(type(model_summary))
    print(type(model.summary()))
    np.savetxt(val_accuracy_on_validation, (history.history['val_acc']), fmt='%.18e', delimiter=' ', newline=os.linesep)
    np.savetxt(val_accuracy_on_training, (history.history['acc']), fmt='%.18e', delimiter=' ', newline=os.linesep)
    np.savetxt(val_loss_on_validation, (history.history['val_loss']), fmt='%.18e', delimiter=' ', newline=os.linesep)
    np.savetxt(val_loss_on_training, (history.history['loss']), fmt='%.18e', delimiter=' ', newline=os.linesep)
    # np.savetxt(file_model_sumary , (model_summary), fmt='%.18e',delimiter=' ', newline=os.linesep)
    text_file = open(file_model_sumary, "w")
    text_file.write(model_summary)

    text_file.close()



def get_lstm_feats(a=20000,b=10,c=300,bat=32,seed=42,run=1):
    # return train pred prob and test pred prob
    NUM_WORDS = a
    N = b
    MAX_LEN = c
    NUM_CLASSES = 3

    nameOfFile = 'SimpleRNN 1layer HiddenNeurons '+ str(N) +' RMSprop 0_001 relu L1_1e-05 Run ' + str(run)
    
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

    model.add(SimpleRNN(N, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=regularizers.l1(0.00001),
                        recurrent_regularizer=regularizers.l1(0.00001), bias_regularizer=None, activity_regularizer=regularizers.l1(0.00001),
                        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                        recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False,
                        stateful=False, unroll=False))

    # model.add(Dense(N)) # this is a fully-connected layer with N hidden units.
    #  model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='relu'))

 

    #optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()  # prints a summary representation of your model.

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    modelName = '/home/s1779494/MLP/experiments/results/[model] ' + nameOfFile + '.h5'
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


hiddenNeuron = 20
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)
    
hiddenNeuron = 50
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)
    
hiddenNeuron = 100
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)

hiddenNeuron = 200
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)

hiddenNeuron = 300
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)

hiddenNeuron = 400
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)
    
hiddenNeuron = 500
for i in range(1,6):
    backend.clear_session()
    get_lstm_feats(16000,hiddenNeuron,300,256,seed=42*i,run=i)







