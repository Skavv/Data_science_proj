import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf

def next_batch(x_data,batch_size):
    
    rindx = np.random.choice(x_data.shape[0], batch_size, replace=False)
    x_batch = x_data[rindx,:]
    return x_batch

def readData(path):
    #df = pd.read_csv("Datasets/Breast_cancer/dataR2.csv")
    #print(df)
    df = pd.read_csv(path)

    # reshuffle the data
    df=df.sample(frac=1).reset_index(drop=True)

    #-----------Checking target class------------
    count_classes = pd.value_counts(df['Classification'], sort = True).sort_index()
    count_classes.plot(kind = 'bar')
    plt.title("Cancer class histogram")
    plt.xlabel("Classification")
    plt.ylabel("Frequency")
    #plt.show()
    #Both cancer and non cancer patients are almost equal in size
    #so it doesn't make sense to resample the dataset


    #sample a training set of cancer patients
    cancer_indices = np.array(df[df.Classification == 2].index)
    number_records_cancer = len(cancer_indices)

    # Picking the indices of the normal classes
    normal_indices = np.array(df[df.Classification == 1].index)
    number_records_normal = len(normal_indices)

    print(number_records_cancer)
    print(number_records_normal)

    #Split 70% training data
    trainingratio = 0.7
    training_n_normal = round(number_records_normal*trainingratio)
    training_n_cancer = round(number_records_cancer*trainingratio)

    # Select the cancer cases trainingset
    random_cancer_indices = np.random.choice(cancer_indices, training_n_cancer, replace = False)
    random_cancer_indices = np.array(random_cancer_indices)
    #print(random_cancer_indices)

    # Out of the cancer indices pick training_n_normal cases with replacement to oversample
    #duplicated_cancer_indices = np.random.choice(random_cancer_indices, training_n_normal, replace = True)
    #Without oversampling
    #duplicated_cancer_indices = np.random.choice(random_cancer_indices, training_n_normal, replace = False)
    #duplicated_cancer_indices = np.array(duplicated_cancer_indices)

    # Select random training normal cases without replacement
    random_normal_indices = np.random.choice(normal_indices, training_n_normal, replace = False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    #sample_indices = np.concatenate([random_normal_indices,duplicated_cancer_indices])
    sample_indices = np.concatenate([random_normal_indices,random_cancer_indices])
    return sample_indices, df

def scaleData(sample_data,test_data):
    #-----------Scale data using MinMaxScaler----------------
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df.drop(['Classification'],axis=1))

    scaled_data = scaler.transform(sample_data.drop(['Classification'],axis=1))
    scaled_test_data = scaler.transform(test_data.drop(['Classification'],axis=1))
    print("Size training data: ", len(scaled_data))
    print("Size test data:     ", len(scaled_test_data))
    #print(scaled_test_data)
    return scaled_data,scaled_test_data

def defineEncoderandEval(learning_rate, keep_prob, act_func, num_hidden, batch_size):
    #DEFINING THE ENCODER
    num_inputs = len(scaled_data[1])
    #num_hidden = 10
    num_outputs = num_inputs 

    #learning_rate = 0.1
    #keep_prob = 1
    tf.reset_default_graph() 

    #We define the placeholders and the layers. We use a simple model. I tried some other activations 
    # but the tanh gives the best results. Also a dropout layer is defined to force the network to generalize.
    # placeholder X
    X = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # weights
    initializer = tf.variance_scaling_initializer()
    w = tf.Variable(initializer([num_inputs, num_hidden]), dtype=tf.float32)
    w_out = tf.Variable(initializer([num_hidden, num_outputs]), dtype=tf.float32)

    # bias
    b = tf.Variable(tf.zeros(num_hidden))
    b_out = tf.Variable(tf.zeros(num_outputs))

    #activation
    #act_func = tf.nn.softmax

    # layers
    hidden_layer = act_func(tf.matmul(X, w) + b)
    dropout_layer= tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
    output_layer = tf.matmul(dropout_layer, w_out) + b_out

    #Functions
    #The loss and the optimizer have to be defined. For the loss we want the output (output_layer) 
    # to be as close to the input (X) as possible. The maximum value of X is 2.

    #We specifiy a function to create a new batch with random data.

    loss = tf.reduce_mean(tf.abs(output_layer - X))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train  = optimizer.minimize( loss)
    init = tf.global_variables_initializer()

    #Let's define the training and evaluate the loss from each epoch.
    num_steps = 50
    #batch_size = 20
    num_batches = len(scaled_data) // batch_size

    with tf.Session() as sess:
        sess.run(init)
        for step in range(num_steps):        
            for iteration in range(num_batches):
                X_batch = next_batch(scaled_data,batch_size)
                sess.run(train,feed_dict={X: X_batch})
            
            if step % 1 == 0:
                err = loss.eval(feed_dict={X: scaled_data})
                if(step == 49):
                    print(step, "\tLoss:", err)
                output_2d = hidden_layer.eval(feed_dict={X: scaled_data})
        
        output_2d_test = hidden_layer.eval(feed_dict={X: scaled_test_data})
    return output_2d

sample_indices, df = readData("Dataset/2.csv")

# Sample dataset
sample_data = df.iloc[sample_indices,:]
test_data = df.drop(sample_indices,axis=0)

print(sample_data)

# sort on Class for the scatter plots at the end, to make sure that Frauds are drawn last
test_data=test_data.sort_values(['Classification'], ascending=[True])

#shuffle the data, because the frauds where added to the tail
sample_data=sample_data.sample(frac=1).reset_index(drop=True)

scaled_data, scaled_test_data = scaleData(sample_data,test_data)

#learning_rate, keep_prob, act_func, num_hidden, batch_size
output_2d = defineEncoderandEval(0.1, 1, tf.nn.softmax, 10, 20)

#The hidden layer is trained, let's see where the frauds (yellow) and non-frauds are located in the 2 dimensional space.
plt.figure()
plt.scatter(output_2d[:,0],output_2d[:,1],c=sample_data['Classification'],alpha=0.7)

#plt.figure()
#plt.scatter(output_2d_test[:,0],output_2d_test[:,1],c=test_data['Classification'],alpha=1)
plt.show()