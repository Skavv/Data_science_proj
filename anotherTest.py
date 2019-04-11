import numpy as np
import csv
import os

DIR_NAME = 'Dataset'

def loadShuffleCSV(path):
    #-----------Load CSV and shuffle the data------------
    result = np.array(list(csv.reader(open(path, "rt"), delimiter=","))).astype("float32")
    np.random.shuffle(result)
    return result
    #----------------------------------------------------

def normalize(item):
    max = item.max(axis=0)
    min = item.min(axis=0)
    for i in range(len(item)):
        for j in range(len(max)):
            item[i][j] = (item[i][j] - min[j])/(max[j]-min[j])
    return max,min,item
    
def splitTestTrain(item, NUM_ROWS, NUM_COLS):
    x_train, x_test = item[:int(0.7*len(item)),:], item[int(0.7*len(item)):,:]
    #y_train = x_train[:,9:10]
    y_train = x_train[:,NUM_COLS-1:NUM_COLS]
    #x_train = x_train[:,0:9]
    x_train = x_train[:,0:NUM_COLS-1]
    #y_test = x_test[:,9:10]
    y_test = x_test[:,NUM_COLS-1:NUM_COLS]
    #x_test = x_test[:,0:9]
    x_test = x_test[:,0:NUM_COLS-1]
    return x_train, y_train, x_test, y_test

def trainNN(name, autoenc_output_xtrain, ytrain, autoenc_output_xtest, ytest, NUM_ROWS, NUM_COLS):
    #------Using intermediate output to train a standard neural net------
    #Wait 50 epochs for validation loss to improve, saving the best model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(str(name) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    if name==0:#Breast cancer dataset
        visible = Input(shape=(NUM_COLS-1,))
        hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
        hidden2 = Dense(NUM_COLS*2, activation='relu')(hidden1)
        hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error', metrics=['accuracy'])
        model.fit(autoenc_output_xtrain, ytrain,
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest, ytest),
                callbacks=[es, mc])
    elif name==1:#Dermatology dataset
        visible = Input(shape=(NUM_COLS-1,))
        hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
        hidden2 = Dense(NUM_COLS*2, activation='tanh')(hidden1)
        hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error', metrics=['accuracy'])
        model.fit(autoenc_output_xtrain, ytrain,
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest, ytest),
                callbacks=[es, mc])
    #from numpy import genfromtxt
    #my_data = genfromtxt('Dataset/ex/unknownset.csv', delimiter=',')
    #my_data = loadShuffleCSV("Dataset/ex/unknownset.csv")
    #maxTmp, minTmp, my_data = normalize(my_data)
    #s = model.predict(my_data, verbose=1)
    return# s, my_data, maxTmp, minTmp

def fixBalanceDermatology(item):
    import matplotlib.pyplot as plt
    
    newArr = item[:,len(item[0])-1]
    plt.hist(newArr, bins=np.arange(newArr.min(), newArr.max()+1))
    plt.show()
    return

#-----------Open every CSV file in the folder-----------
#1.csv is breast cancer dataset
#2.csv is dermatology
result = []
for filename in os.listdir(DIR_NAME):
    if filename.endswith(".csv"): 
        result.append(loadShuffleCSV(DIR_NAME + "/" + filename))
        continue
    else:
        continue
#-------------------------------------------------------

fixBalanceDermatology(result[1])
'''
#---Normalize based on min,max values and save these values---
max = []
min = []
for i in range(len(result)):
    maxTmp,minTmp,result[i] = normalize(result[i])
    max.append(maxTmp)
    min.append(minTmp)
#------------------------------------------------------------

#-----Split dataset into 70% training and 30% testing data---
#Arrays are 3D. The whole array has every dataset saved. First
#dimension(x_train[0]) is the first dataset, or the breast cancer
#dataset. Second dimension(x_train[0][0]) is the first row of the
#cancer dataset. Third dimension(x_train[0][0][0]) is the first 
#element of the first row of the cancer dataset.
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(len(result)):
    NUM_ROWS = len(result[i])
    NUM_COLS = len(result[i][0])
    x_trainTmp, y_trainTmp, x_testTmp, y_testTmp = splitTestTrain(result[i], NUM_ROWS, NUM_COLS)
    x_train.append(x_trainTmp)
    y_train.append(y_trainTmp)
    x_test.append(x_testTmp)
    y_test.append(y_testTmp)
#---------------------------------------------------------

intermediate_layer_model = []
#---------------Defining autoencoder------------------
#Each dataset has a different autoencoder
from keras.layers import Input, Dense, Conv2D, Flatten, Activation
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

for k in range(len(result)):
    NUM_ROWS = len(x_train[k])
    NUM_COLS = len(x_train[k][0])
    inputs = Input(shape=(NUM_COLS, ))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    if k==0:#Breast cancer dataset
        h1 = Dense(128, activation='relu')(inputs)
        h = Dense(64, activation='sigmoid')(h1)
        outputs = Dense(NUM_COLS)(h)
        model = Model(input=inputs, output=outputs)
        print("Breast cancer dataset.")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x_train[k], x_train[k],
            batch_size=5,
            epochs=50,
            validation_data=(x_test[k], x_test[k]),
            callbacks=[es])
        intermediate_layer_model.append(Model(inputs=model.input,
                                 outputs=model.output))
    elif k==1:#Dermatology dataset
        h1 = Dense(512, activation='relu')(inputs)
        h2 = Dense(256, activation='relu')(h1)
        h = Dense(128, activation='sigmoid')(h2)
        outputs = Dense(NUM_COLS)(h)
        model = Model(input=inputs, output=outputs)
        print("Dermatology dataset")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x_train[k], x_train[k],
            batch_size=80,
            epochs=500,
            validation_data=(x_test[k], x_test[k]),
            callbacks=[es])
        intermediate_layer_model.append(Model(inputs=model.input,
                                 outputs=model.output))


#------------------Predict values-------------------
autoenc_output_x_test = []
autoenc_output_x_train = []

for i in range(len(result)):
    autoenc_output_x_test.append(intermediate_layer_model[i].predict(x_test[i]))
    autoenc_output_x_train.append(intermediate_layer_model[i].predict(x_train[i]))

my_data = []
for i in range(len(result)):
    NUM_ROWS = len(result[i])
    NUM_COLS = len(result[i][0])
    #s, my_data, maxTmp, minTmp = trainNN(i, x_train[i], y_train[i], autoenc_output_x_train[i], y_test[i], NUM_ROWS, NUM_COLS)
    trainNN(i, autoenc_output_x_train[i], y_train[i], autoenc_output_x_test[i], y_test[i], NUM_ROWS, NUM_COLS)
    maxTmp = np.array(maxTmp)
    minTmp = np.array(minTmp)
    print(maxTmp.shape)

#-------------De-normalizing values--------------
for i in range(len(autoenc_output_x_test)):
    for j in range(len(autoenc_output_x_test[i])):
        for k in range(len(autoenc_output_x_test[i][j])):
            autoenc_output_x_test[i][j][k] = (autoenc_output_x_test[i][j][k]*(max[i][k]-min[i][k]))+min[i][k]
            x_test[i][j][k] = (x_test[i][j][k]*(max[i][k]-min[i][k])) + min[i][k]
            x_train[i][j][k] = (x_train[i][j][k]*(max[i][k]-min[i][k])) + min[i][k]
            if j<11:
                my_data[j][k] = (my_data[j][k]*(maxTmp[k]-minTmp[k])) + minTmp[k]
            if k==8:
                y_train[i][j][0] =  (y_train[i][j][0]*(max[i][k+1]-min[i][k+1])) + min[i][k+1]
                y_test[i][j][0] =  (y_test[i][j][0]*(max[i][k+1]-min[i][k+1])) + min[i][k+1]

#print(x_test[0][1])
#print(autoenc_output_x_test[0][1])
#print(y_test[0][1])
#------------------------------------------------


for i in range(len(s)):
    s[i] = s[i]+1
    print(str(s[i])+" "+str(my_data[i]))
#print(s)



'''