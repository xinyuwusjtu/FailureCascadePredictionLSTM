from pandas import DataFrame
from pandas import concat
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import time
import xlwt

def save(data, path):
    f = xlwt.Workbook()  
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  
    [h, l] = data.shape  
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)

np.set_printoptions(suppress=True)

SST=np.load("SST_30_FailSize_4_FixedCapacity_0_8_AC.npy")
SSV=np.load("SSV_30_FailSize_4_FixedCapacity_0_8_AC.npy")

M=len(SST[:,1,1])
Q=len(SST[1,:,1])
K=len(SST[1,1,:])

### Load temporarily saved model and weights
##json_file = open('model_IEEE30_FailSize_4_FixedCapacity_0_8_AC_Neuron_300_ErrorEachRound.json', 'r')
##loaded_model_json = json_file.read()
##json_file.close()
##model = model_from_json(loaded_model_json)
### load weights into new model
##model.load_weights("model_IEEE30_FailSize_4_FixedCapacity_0_8_AC_Neuron_300_ErrorEachRound.h5")
##print("Loaded model from disk")

##Design neural network
n_batch = 1
n_epoch = 100
n_neurons = 300
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, M), stateful=False, return_sequences=True,))
model.add(Dense(M,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

avg_err=[];
index=[];

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.ion()
plt.show()

Training_size=4900
Test_size=100
##Set the training set size and the test set size
for k in range(0,Training_size):
    sequence = SST[:,:,k]
    df = np.zeros((Q-1,M,2))
    for j in range(Q-1):
        df[j,:,0] = sequence[:,j]
        df[j,:,1] = sequence[:,j+1]
    X, y = df[:,:,0], df[:,:,1]
    for j in range(int(SSV[k])):
        X_i=np.zeros(M)
        y_i=np.zeros(M)
        for i in range(M):
            X_i[i]=X[j,i]
            y_i[i]=y[j,i]
        X_i = X_i.reshape(1, 1, M)
        y_i = y_i.reshape(1, 1, M)
        for _ in range(n_epoch):
            model.fit(X_i, y_i, epochs=1, batch_size=1, verbose=0, callbacks=None, shuffle=False)
            model.reset_states()

    ##Examine the predicting performance when every 5 more sampled sequences are applied to train
    if k % 5 ==0:
        error_cum=0
        for q in range(Test_size):
            for t in range(int(SSV[Training_size+q]-1)):
                sequence_x=np.zeros(M)
                sequence_y=np.zeros(M)
                for p in range(M):
                    sequence_x[p] = SST[p,t,Training_size+q]
                    sequence_y[p] = SST[p,t+1,Training_size+q]
                testX, testy = sequence_x, sequence_y
                testX_tmp = testX.reshape(1, 1, M)
                yhat = model.predict(testX_tmp, batch_size=1,verbose=0)
            yhat=yhat.reshape(M)
            for t in range(len(yhat)):
                if yhat[t]>=0.5:
                    yhat[t]=1
                else: yhat[t]=0
            error_cum=error_cum+np.sum(np.square(testy-yhat))                  
        err=error_cum/Test_size

        ##Draw the average predicting error 
        x_axis=np.append(index,k)
        y_axis=np.append(avg_err,err)
        ax.scatter(x_axis, y_axis)
        plt.pause(0.5)

    ##Save temporarily learned model to json timely
    if k % 500 == 0:
        model_json = model.to_json()
        with open("model_IEEE30_FailSize_5_FixedCapacity_0_8_AC_Neuron_300_ErrorEachRound.json","w") as json_file:
            json_file.write(model_json)

        ##Save weights to h5
        model.save_weights("model_IEEE30_FailSize_4_FixedCapacity_0_8_AC_Neuron_300_ErrorEachRound.h5")

