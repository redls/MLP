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
from collections import OrderedDict
import pickle


train_df = pd.read_csv("/home/s1779494/MLP/experiments/train.csv")
test_df = pd.read_csv("/home/s1779494/MLP/experiments/test.csv")



def train_model_and_plot_stats(history,nameOfFile=" ", path=" ", model=None):
    lossFileName = path + '/' + nameOfFile + ' loss.pdf';
    accuracyFileName = path + '/' + nameOfFile + ' accuracy.pdf';
    val_accuracy_on_validation = path + '/' + nameOfFile + ' accuracy on validation set.txt';
    file_model_sumary = path + '/' + nameOfFile + ' model sumary.txt';
    val_accuracy_on_training = path + '/' + nameOfFile + ' accuracy on training set.txt'
    val_loss_on_validation = path + '/' + nameOfFile + ' loss on validation set.txt'
    val_loss_on_training = path + '/' + nameOfFile + ' loss on training set.txt'
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
    plt.close(fig_1)
    plt.close(fig_2)


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
    with open(path+'/HISTORY_'+nameOfFile+'.pkl', 'wb') as f:
        pickle.dump(history.history, f)



def get_lstm_feats(a=20000,b=10,c=300,bat=32,seed=42,run=1,opt=""):
    # return train pred prob and test pred prob
    NUM_WORDS = a
    N = b
    MAX_LEN = c
    NUM_CLASSES = 3
    nameOfFile = 'SimpleRNN 1layer '+opt+' relu Run ' + str(run)
    name = '/home/s1779494/MLP/experiments/results/' + opt
    if not os.path.exists(name):
        os.makedirs(name)
        path = name 
    else:
        path = name 
            
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

    
    opt_splitted = opt.split()
    opt_name = opt_splitted[0]
    learning__Rate = float(opt_splitted[1])
    
    if opt_name == 'SGD':
        optim = optimizers.SGD(lr=learning__Rate, momentum=0.0, decay=0.0, nesterov=False)
    elif opt_name == 'RMSprop':
        optim = optimizers.RMSprop(lr=learning__Rate, rho=0.9, epsilon=None, decay=0.0)
    elif opt_name == 'Adam':
        optim = optimizers.Adam(lr=learning__Rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif opt_name == 'Adagrad':
        optim = optimizers.Adagrad(lr=learning__Rate, epsilon=None, decay=0.0)
     
    model = Sequential()
    model.add(Embedding(NUM_WORDS, N, input_length=MAX_LEN))
    model.add(SimpleRNN(N, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                        recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False,
                        stateful=False, unroll=False))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.summary()  # prints a summary representation of your model.

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    modelName = path + '/[model] ' + nameOfFile + '.h5'
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
    train_model_and_plot_stats(history, nameOfFile, path, model=model)

    #     model = load_model(modelName)
    #     train_pred = model.predict(train_x)
    #     test_pred = model.predict(test_x)
    del model
    gc.collect()
    return history
#     print(log_loss(train_y,train_pred))
#     return train_pred,test_pred




# main body
for i in range(1,6):
    optimisers = ['SGD 0.01', 'SGD 0.001', 'SGD 0.0001', 'SGD 0.00001', 
                  'RMSprop 0.01', 'RMSprop 0.001', 'RMSprop 0.0001', 'RMSprop 0.00001',
                  'Adam 0.01', 'Adam 0.001', 'Adam 0.0001', 'Adam 0.00001', 
                  'Adagrad 0.01', 'Adagrad 0.001', 'Adagrad 0.0001', 'Adagrad 0.00001']  
    run_info = OrderedDict()
    for optimiser in optimisers: 
        backend.clear_session()
        run_info[optimiser] = get_lstm_feats(16000,12,300,256,seed=42*i,run=i,opt=optimiser)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for optimiser, history in run_info.items():
        print('optimiser: ',optimiser)
        print('history: ',history)
        ax1.plot(history.history['val_acc'], label=str(optimiser))
        ax2.plot(history.history['val_loss'], label=str(optimiser))
    ax1.legend(loc=0)
    ax1.set_xlabel('Epoch number')
    ax1.set_ylabel('Validation set accuracy') 
    ax2.legend(loc=0)
    ax2.set_xlabel('Epoch number')
    ax2.set_ylabel('Validation set error') 

    fig.tight_layout() 
    fig.savefig('/home/s1779494/MLP/experiments/results/ALL_optimisers_learningRates Run '+str(i)+'.pdf')


