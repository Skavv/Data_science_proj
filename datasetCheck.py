'''
'This is to check the datasets
'''
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

DIR_NAME = 'Dataset'
def fixDatasets(item, i):
    if i==0:#Cancer dataset
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([0,3,0,100])
        plt.savefig('cancerDataset.png')
        #plt.show()
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
        #plt.show()
    elif i==2:
        newArr = item[:,len(item[0])-1].astype("int64")
        plt.bar(np.arange(newArr.min(), newArr.max()+1), np.bincount(newArr)[1:])
        plt.axis([-1,2,0,500])

        plt.savefig('auditDataset.png')
        #plt.show()
    return
#-----------Open every CSV file in the folder-----------
#1.csv is breast cancer dataset
#2.csv is dermatology
#3.csv is audit_risk
result = []
for filename in os.listdir(DIR_NAME):
    if filename.endswith(".csv"): 
        arr = np.array(list(csv.reader(open(DIR_NAME + "/" + filename, "rt"), delimiter=","))).astype("float32")
        result.append(arr)
        continue
    else:
        continue
#-------------------------------------------------------

for i in range(len(result)):
    fixDatasets(result[i], i)

#result[0] = 
#fixCancerDataset(result[0])
#result[1] = fixBalanceDermatology(result[1])
#fixBalanceDermatology(result[1])
#fixBalanceAudit(result[2])