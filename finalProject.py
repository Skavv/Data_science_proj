import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, RMSprop
from sklearn import metrics

DIR_NAME = 'Dataset'#Folder name in which the datasets are located
PERCENTAGE = 100#Percentage of the data to be used

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

def fixDatasets(item, i):
    if i==0:#Cancer dataset
        #Visual of breast cancer dataset saved as .png file
        #within the folder
        '''
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,3,0,100])
        plt.savefig('cancerDataset.png')
        plt.show()
        '''
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
        
        #Visual of edited dataset saved as .png file in the folder
        '''
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,7,0,200])
        plt.savefig('dermatologyDatasetEdited.png')
        #plt.show()
        '''
        return df.values
    if i==2:
        #Visual of the audit dataset saved as .png file in the folder
        '''
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([-1,2,0,500])
        plt.savefig('auditDataset.png')
        #plt.show()
        '''
        return item
    return

def splitTestTrain(item, NUM_ROWS, NUM_COLS, i):
    item = item[:int(len(item) * (PERCENTAGE/100))]
    unique, counts = np.unique(item[:,len(item[0])-1], return_counts=True)
    x_train, x_test = item[:int(0.7*len(item)),:], item[int(0.7*len(item)):,:]
    initial = counts[0]/counts[1]
    #Making sure the datasets are equally splitted to training and test data
    #according to the initial dataset
    if i==0 or i==2:#Breast cancer dataset and audit dataset
        unique, counts = np.unique(x_train[:,len(x_train[0])-1], return_counts=True)
        second = counts[0]/counts[1]
        while initial+0.05<second or initial-0.05>second:
            np.random.shuffle(item)
            x_train, x_test = item[:int(0.7*len(item)),:], item[int(0.7*len(item)):,:]
            unique, counts = np.unique(x_train[:,len(x_train[0])-1], return_counts=True)
            second = counts[0]/counts[1]
    elif i==1:#Dermatology dataset
        occurence_perc = []
        for j in range(len(counts)):
            occurence_perc.append(counts[j]/counts.sum())
        unique, counts = np.unique(x_train[:,len(x_train[0])-1], return_counts=True)
        flag = False
        while False:
            if flag==True:
                break
            np.random.shuffle(item)
            x_train, x_test = item[:int(0.7*len(item)),:], item[int(0.7*len(item)):,:]
            unique, counts = np.unique(x_train[:,len(x_train[0])-1], return_counts=True)
            for j in range(len(counts)):
                if counts[j]/counts.sum()+0.01<occurence_perc[j] or counts[j]/counts.sum()-0.01>occurence_perc[j]:
                    break
                flag = True

    y_train = x_train[:,NUM_COLS-1:NUM_COLS]
    x_train = x_train[:,0:NUM_COLS-1]
    y_test = x_test[:,NUM_COLS-1:NUM_COLS]
    x_test = x_test[:,0:NUM_COLS-1]
    return x_train, y_train, x_test, y_test

def trainNN(name, autoenc_output_xtrain, ytrain, autoenc_output_xtest, ytest, NUM_ROWS, NUM_COLS):
    #------Using autoencoder hidden layer output to train a standard neural net------
    #Wait 100 epochs for validation loss to improve, saving the best model
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
    mc = ModelCheckpoint(r'./NN_weights/'+str(name) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    if name==0:#Breast cancer dataset
        visible = Input(shape=(NUM_COLS,))
        hidden1 = Dense(4, activation='relu')(visible)
        output = Dense(1, activation='sigmoid')(hidden1)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=SGD(lr=0.4, decay=1e-6, momentum=0.001, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
        model.fit(autoenc_output_xtrain[name], ytrain[name],
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest[name], ytest[name]),
                callbacks=[es, mc])
        model = load_model(r'./NN_weights/' + str(name) + '.h5')
        pred = model.predict(autoenc_output_xtest[name])
        for i in range(len(pred)):
            if pred[i]>0.5:
                pred[i] = 1
            elif pred[i]<=0.5:
                pred[i] = 0
    elif name==1:#Dermatology dataset
        visible = Input(shape=(NUM_COLS,))
        hidden1 = Dense(16, activation='relu')(visible)
        output = Dense(1, activation='sigmoid')(hidden1)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.4, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
        model.fit(autoenc_output_xtrain[name], ytrain[name],
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest[name], ytest[name]),
                callbacks=[es, mc])   
    elif name==2:#Audit dataset
        visible = Input(shape=(NUM_COLS,))
        hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
        hidden2 = Dense(NUM_COLS*2, activation='tanh')(hidden1)
        hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
        output = Dense(1, activation='sigmoid')(hidden3)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
        model.fit(autoenc_output_xtrain[name], ytrain[name],
                batch_size=15,
                epochs=500,
                validation_data=(autoenc_output_xtest[name], ytest[name]),
                callbacks=[es, mc])
    return

class AutoEncoder:
    def __init__(self, encoding_dim, arr):
        self.encoding_dim = encoding_dim
        self.x = np.array(arr)

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self, col):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(col)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self, col):
        ec = self._encoder()
        dc = self._decoder(col)
        
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        
        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        self.model.fit(self.x, self.x,
                        epochs=epochs,
                        batch_size=batch_size)

    def save(self,i):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights'+str(i)+'.h5')

if __name__ == '__main__':
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

    #-----Split datasets into 70% training and 30% testing data---
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
        x_trainTmp, y_trainTmp, x_testTmp, y_testTmp = splitTestTrain(result[i], NUM_ROWS, NUM_COLS, i)
        x_train.append(x_trainTmp)
        y_train.append(y_trainTmp)
        x_test.append(x_testTmp)
        y_test.append(y_testTmp)
    #---------------------------------------------------------
    
    #Create an autoencoder for each of the datasets reducing the
    #dimension
    for i in range(len(result)):
        if i==0:#Breast cancer dataset
            ae = AutoEncoder(encoding_dim=5, arr=x_train[i])
            ae.encoder_decoder(col=len(x_train[i][0]))
            ae.fit(batch_size=10, epochs=500)
            ae.save(i=i)        
        elif i==1:#Dermatology dataset
            ae = AutoEncoder(encoding_dim=17, arr=x_train[i])
            ae.encoder_decoder(col=len(x_train[i][0]))
            ae.fit(batch_size=20, epochs=500)
            ae.save(i=i)
        if i==2:#Audit dataset
            ae = AutoEncoder(encoding_dim=12, arr=x_train[i])
            ae.encoder_decoder(col=len(x_train[i][0]))
            ae.fit(batch_size=50, epochs=500)
            ae.save(i=i)
    #------------------------------------------------------------
    
    #Reduce dimensionality of datasets and create a new array
    #reduced_Arr[i] is the reduced x_(test,train)[i] list 
    #and y_(test, train)[i] is it's output which the neural network
    #will later be trained on
    reduced_Arr_x_train = []
    reduced_Arr_x_test = []
    for i in range(len(x_train)):
        encoder = load_model(r'./weights/encoder_weights'+str(i)+'.h5')
        tmp = encoder.predict(x_train[i])
        tmp2 = encoder.predict(x_test[i])
        reduced_Arr_x_train.append(tmp)
        reduced_Arr_x_test.append(tmp2)
    #------------------------------------------------------------

    #------Train a neural network based on the reduced dataset-------
    for i in range(len(reduced_Arr_x_train)):
        NUM_ROWS = len(reduced_Arr_x_train[i])
        NUM_COLS = len(reduced_Arr_x_train[i][0])
        trainNN(i, reduced_Arr_x_train, y_train, reduced_Arr_x_test, y_test, NUM_ROWS, NUM_COLS)
        maxTmp = np.array(maxTmp)
        minTmp = np.array(minTmp)
    #----------------------------------------------------------------

    #Train 3 separate neural networks to compare the accuracy scores-----------------------------------
    #to the accuracy scores of the reduced dataset training--------------------------------------------
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
    mc = ModelCheckpoint(r'./notReducedNN_weights/'+str(0) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    #Breast cancer dataset
    visible = Input(shape=(len(result[0][0])-1,))
    hidden1 = Dense(4, activation='relu')(visible)
    output = Dense(1, activation='sigmoid')(hidden1)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer=SGD(lr=0.4, decay=1e-6, momentum=0.001, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train[0], y_train[0],
                batch_size=15,
                epochs=500,
                validation_data=(x_test[0], y_test[0]),
                callbacks=[es, mc])
    #Dermatology dataset
    mc = ModelCheckpoint(r'./notReducedNN_weights/'+str(1) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    visible = Input(shape=(len(result[1][0])-1,))
    hidden1 = Dense(16, activation='relu')(visible)
    output = Dense(1, activation='sigmoid')(hidden1)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.4, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train[1], y_train[1],
                batch_size=15,
                epochs=500,
                validation_data=(x_test[1], y_test[1]),
                callbacks=[es, mc])
    #Audit dataset
    mc = ModelCheckpoint(r'./notReducedNN_weights/'+str(2) + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    visible = Input(shape=(len(result[2][0])-1,))
    hidden1 = Dense(NUM_COLS-1, activation='relu')(visible)
    hidden2 = Dense(NUM_COLS*2, activation='tanh')(hidden1)
    hidden3 = Dense(NUM_COLS-1, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train[2], y_train[2],
                batch_size=15,
                epochs=500,
                validation_data=(x_test[2], y_test[2]),
                callbacks=[es, mc])
    #---------------------------------------------------------------------------------------------------
    
    #Code to visualize an encoded and decoded entry of--------------------------------------------------
    #the breast cancer dataset--------------------------------------------------------------------------
    '''
    encoder = load_model(r'./weights/encoder_weights0.h5')
    decoder = load_model(r'./weights/decoder_weights0.h5')
    inputs = np.array([[0.61538464,0.79963964,0.24822696,0.0356072,0.02318621,0.19659062,0.10462296,0.04426264,0.125019]])
    x = encoder.predict(inputs)
    y = decoder.predict(x)

    print('Input: {}'.format(inputs))
    print('Encoded: {}'.format(x))
    print('Decoded: {}'.format(y))
    '''
    #-----------------------------------------------------------------------------------------------------