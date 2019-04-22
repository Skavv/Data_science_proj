import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

DIR_NAME = 'Dataset'
PERCENTAGE = 100

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
    item = item[:int(len(item) * (PERCENTAGE/100))]
    x_train, x_test = item[:int(0.7*len(item)),:], item[int(0.7*len(item)):,:]
    y_train = x_train[:,NUM_COLS-1:NUM_COLS]
    x_train = x_train[:,0:NUM_COLS-1]
    y_test = x_test[:,NUM_COLS-1:NUM_COLS]
    x_test = x_test[:,0:NUM_COLS-1]
    return x_train, y_train, x_test, y_test

def trainNN(name, autoenc_output_xtrain, ytrain, autoenc_output_xtest, ytest, NUM_ROWS, NUM_COLS):
    #------Using intermediate output to train a standard neural net------
    #Wait 50 epochs for validation loss to improve, saving the best model
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    es = EarlyStopping(monitor='auc_roc', patience=50, verbose=1, mode='max')
    #mc = ModelCheckpoint(str(name) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    mc = ModelCheckpoint(str(name) + '.h5', monitor='auc_roc', mode='max', verbose=1, save_best_only=True)
    if name==0:#Breast cancer dataset
        visible = Input(shape=(NUM_COLS-1,))
        hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
        hidden2 = Dense(NUM_COLS*2, activation='relu')(hidden1)
        hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error', metrics=['accuracy', auc_roc])
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
        model.compile(optimizer=RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error', metrics=['accuracy', auc_roc])
        model.fit(autoenc_output_xtrain, ytrain,
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest, ytest),
                callbacks=[es, mc])
    elif name==2:#Audit dataset
        visible = Input(shape=(NUM_COLS-1,))
        hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
        hidden2 = Dense(NUM_COLS*2, activation='tanh')(hidden1)
        hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=RMSprop(lr=0.5, rho=0.9, epsilon=None, decay=0.0), loss='mean_squared_error', metrics=['accuracy', auc_roc])
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

def fixDatasets(item, i):
    if i==0:#Cancer dataset
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,3,0,100])
        plt.savefig('cancerDataset.png')
        #plt.show()
        return item
    elif i==1:#Dermatology dataset
        '''
        #Visual of excessive entries
        
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,7,0,200])

        plt.savefig('dermatologyDataset.png')
        plt.show()
        '''

        df = pd.DataFrame(data=item)

        #Too many samples with "1" output
        #Remove some rows to make it equal
        #to the other column values
        a = df.loc[df[len(item[0])-1] == 1]
        a.sample(frac=1).reset_index(drop=True)
        a = a.iloc[40:,]
        
        #Remove already existing rows with 
        #column value 1 from the dataset
        df = df[df[len(item[0])-1] != 1]
        
        #Re-inserting them and shuffling
        #the dataset
        df = pd.concat([df, a])
        df = df.sample(frac=1).reset_index(drop=True)

        #Too few samples with "6" output
        #Duplicate random rows to make it equal
        #There are only 20 instances
        a = df.loc[df[len(item[0])-1] == 6]
        a.sample(frac=1).reset_index(drop=True)
        

        #Insert duplicated rows and shuffle
        #the dataset
        df = pd.concat([df, a])
        df = pd.concat([df, a])
        df = df.sample(frac=1).reset_index(drop=True)

        #Printing the values again after undersampling
        #print(len(df.columns))
        newArr = df.values[:,len(df.columns)-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,7,0,200])
        plt.savefig('dermatologyDatasetEdited.png')
        return df.values
        #plt.show()
    elif i==2:
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([-1,2,0,500])

        plt.savefig('auditDataset.png')
        #plt.show()
        return item
    return

def auc_roc(y_true, y_pred):
    import tensorflow as tf
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


#-----------Open every CSV file in the folder-----------
#1.csv is breast cancer dataset
#2.csv is dermatology
#3.csv is audit_risk
result = []
for filename in os.listdir(DIR_NAME):
    if filename.endswith(".csv"): 
        result.append(loadShuffleCSV(DIR_NAME + "/" + filename))
        continue
    else:
        continue
#-------------------------------------------------------

#Check all the datasets and fix the 
#Dermatology dataset
for i in range(len(result)):
    result[i] = fixDatasets(result[i], i)

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
from keras.layers import Input, Dense, Conv1D, Flatten, Activation
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
    mc = ModelCheckpoint('autoenc_' + str(k) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
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
            callbacks=[mc])
        intermediate_layer_model.append(Model(inputs=model.input,
                                 outputs=model.output))
    elif k==1:#Dermatology dataset
        h1 = Dense(20, activation='relu')(inputs)
        h2 = Dense(100, activation='relu')(h1)
        h3 = Dense(20, activation='sigmoid')(h2)
        #outputs = Dense(NUM_COLS)(h3)
        outputs = Dense(NUM_COLS)(h3)
        model = Model(input=inputs, output=outputs)
        print("Dermatology dataset")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x_train[k], x_train[k],
            batch_size=15,
            epochs=500,
            validation_data=(x_test[k], x_test[k]),
            callbacks=[mc])
        intermediate_layer_model.append(Model(inputs=model.input,
                                 outputs=model.output))
    elif k==2:#audit_risk dataset
        h1 = Dense(20, activation='relu')(inputs)
        h2 = Dense(10, activation='sigmoid')(h1)
        outputs = Dense(NUM_COLS)(h2)
        model = Model(input=inputs, output=outputs)
        print("Audit risk dataset")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x_train[k], x_train[k],
            batch_size=10,
            epochs=500,
            validation_data=(x_test[k], x_test[k]),
            callbacks=[mc])
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

'''
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


