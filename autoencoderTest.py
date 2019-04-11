import keras
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
encoding_dim = 3  # 3 floats -> compression of factor 3, assuming the input is 9 floats

# this is our input placeholder
input_img = Input(shape=(9,))


#--------------DENSE-----------------------------
encoded = Dense(6, activation='relu')(input_img)
encoded = Dense(3, activation='relu')(encoded)

decoded = Dense(3, activation='relu')(encoded)
decoded = Dense(9, activation='linear')(decoded)
autoencoder = Model(input_img, decoded)
#-------------------------------------------------
'''
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(9, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
'''
autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
, loss='mean_squared_error')

import csv
result = np.array(list(csv.reader(open("Datasets/Breast_cancer/dataR2.csv", "rt"), delimiter=","))).astype("float32")
np.random.shuffle(result)
x_train, x_test = result[:int(0.7*len(result)),:], result[int(0.7*len(result)):,:]
x_train = x_train[:,0:9]
x_test = x_test[:,0:9]


from keras.datasets import mnist
#import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=50,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.predict(x_test)
decoded_imgs = autoencoder.predict(encoded_imgs)

encoded_imgs = encoded_imgs*255.
decoded_imgs = decoded_imgs*255.
print(encoded_imgs[0]) 
print(decoded_imgs[0])
'''
import numpy as np
import pandas as pd
import tensorflow as tf

#loading the images
all_images = np.loadtxt('Datasets/Breast_cancer/dataR2.csv',\
                  delimiter=',', skiprows=1)[:,0:9]
#looking at the shape of the file
print(all_images.shape)

n_nodes_inpl = 9  #encoder
n_nodes_hl1  = 2  #encoder
n_nodes_hl2  = 2  #decoder
n_nodes_outl = 9  #decoder

# first hidden layer has 9*2 weights and 2 biases
hidden_1_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))  }

# second hidden layer has 2*2 weights and 2 biases
hidden_2_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))  }

# second hidden layer has 2*9 weights and 9 biases
output_layer_vals = {
'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_outl])),
'biases':tf.Variable(tf.random_normal([n_nodes_outl])) }

# image with shape 9 goes in
input_layer = tf.placeholder('float', [None, 9])

# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(
       tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
       hidden_1_layer_vals['biases']))
# multiply output of layer_1 wth a weight matrix and add biases
layer_2 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
       hidden_2_layer_vals['biases']))
# multiply output of layer_2 wth a weight matrix and add biases
output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, 9])

# define our cost function
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))
# define our optimizer
learn_rate = 0.4   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# defining batch size, number of epochs and learning rate
batch_size = 25  # how many images to use together for training
hm_epochs =1000    # how many times to go through the entire dataset
tot_images = 116 # total number of images

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq],\
               feed_dict={input_layer: epoch_x, \
               output_true: epoch_x})
        epoch_loss += c
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)

# pick any image
any_image = all_images[1]
# run it though the autoencoder
output_any_image = sess.run(output_layer,\
                   feed_dict={input_layer:[any_image]})
# run it though just the encoder
encoded_any_image = sess.run(layer_1,\
                   feed_dict={input_layer:[any_image]})
# print the original image
print(all_images[1])
#plt.show()
# print the encoding
print(output_any_image)


#model.fit(x_train,x_train,verbose=1,epochs=10,batch_size=256)
#model.save('auto_en.h5')

# Reshape and normalize training data
df = pd.read_csv("Datasets/Breast_cancer/dataR2.csv").values
shuffle(df)
print(df.shape)
#df = np.reshape(df, df.shape + (1,))  
#print(df.shape)

X = df[:,0:9]/255.0
Y = df[:,9]
print(X.shape)
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

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam


x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(60000,738)

model = Sequential()
model.add(Dense(738,activation='relu',input_dim=738))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(738,activation='relu'))

model.compile(loss=keras.losses.mean_squared_error,
             optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0),
             metrics = ['accuracy'])

model.fit(x_train,x_train,verbose=1,epochs=10,batch_size=256)
model.save('auto_en.h5')
'''