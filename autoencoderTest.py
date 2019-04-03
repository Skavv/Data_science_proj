import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from numpy.random import shuffle
'''
df = pd.read_csv("Datasets/Breast_cancer/dataR2.csv")
# reshuffle the data
df=df.sample(frac=1).reset_index(drop=True)

#sample a training set of cancer patients
cancer_indices = np.array(df[df.Classification == 2].index)
number_records_cancer = len(cancer_indices)

# Picking the indices of the normal classes
normal_indices = np.array(df[df.Classification == 1].index)
number_records_normal = len(normal_indices)

#Split 70% training data
trainingratio = 0.7
training_n_normal = round(number_records_normal*trainingratio)
training_n_cancer = round(number_records_cancer*trainingratio)

# Select the cancer cases trainingset
random_cancer_indices = np.random.choice(cancer_indices, training_n_cancer, replace = False)
random_cancer_indices = np.array(random_cancer_indices)

# Select random training normal cases without replacement
random_normal_indices = np.random.choice(normal_indices, training_n_normal, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
#sample_indices = np.concatenate([random_normal_indices,duplicated_cancer_indices])
sample_indices = np.concatenate([random_normal_indices,random_cancer_indices])

sample_data = df.iloc[sample_indices,:]
test_data = df.drop(sample_indices,axis=0)

#shuffle the data, because the frauds where added to the tail
sample_data=sample_data.sample(frac=1).reset_index(drop=True)
'''


# Reshape and normalize training data
df = pd.read_csv("Datasets/Breast_cancer/dataR2.csv").values
shuffle(df)
print(df.shape)
df = np.reshape(df, df.shape + (1,))  
print(df.shape)

X = df[:,0:9]/255.0
Y = df[:,9]

counter = 0.7*len(X)
x_test = []
y_test = []
x_train = []
y_train = []

for x in range(len(X)):
    if(x>counter):
        x_test.append(X[x])
        y_test.append(Y[x])
    else:
        x_train.append(X[x])
        y_train.append(Y[x])

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
#y_train = lb.fit_transform(y_train)
#y_test = lb.fit_transform(y_test)

model = keras.models.Sequential()
#K.set_image_dim_ordering('th')
model.add(keras.layers.Convolution2D(30, 5, 5, input_shape=(116, 10, 1),activation= 'relu' ))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Convolution2D(15, 3, 3, activation= 'relu' ))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation= 'relu' ))
model.add(keras.layers.Dense(50, activation= 'relu' ))
model.add(keras.layers.Dense(9, activation= 'softmax' ))
# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


#----------------------------------------------------------
