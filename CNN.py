from turtle import end_fill
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D,LeakyReLU
import matplotlib.pyplot as plt
from mat2json import loadMat
from sklearn.model_selection import train_test_split

#Ambient temp 24
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')

def extract_discharge(Battery): 
    cap = [] 
    i = 1
    for Bat in Battery: 
        if Bat['cycle'] == 'discharge':
            cap.append((Bat['data']['Capacity'][0])) 
            i+=1
    return cap

def extract_charge(Battery): 
    a = [] 
    i = 1
    for Bat in Battery: 
        # if i%2==0:
        if Bat['cycle'] == 'discharge':
            a.append((Bat['data']['Capacity'][0]))
            # a[i] = Bat.cycle(i).data.Capacity;
            # a[i] = Bat['data']['Capacity'][0]
            i+=1

    return a


A = extract_charge(B0005) # Get the value of A
InitC = 1.85;# get the value of InitC
cap5 = extract_discharge(B0005); # Get the cap5 

# Scaling the data
def minmax_norm(A,InitC,cap):
    r = np.max(A) - np.min(A); 
    xData = (A - np.min(A))/r; 
    comp = len(A) - len(cap);
    yData = np.vstack((InitC*np.ones((comp, 1)), np.reshape(cap5, (len(cap5), 1)))) 
    ym = np.min(yData);
    yr = np.max(yData) - np.min(yData);
    yData = (yData - ym)/yr;
    return xData, yData, ym, yr       

xData, yData, ym, yr = minmax_norm(A,InitC,cap5)
X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size = 0.20,shuffle = False)# split the data into train and test

# Building the model using Conv2D
input_shape = (1,30,1)
model = Sequential()
model.add(Conv2D(30, kernel_size=(1, 2), strides=(1, 1), input_shape=input_shape))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(10, kernel_size=(1, 2),strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
print(model.summary())

model.compile(loss='mean_squared_error',optimizer='adam',  metrics = ['accuracy'])
model.fit(X_train,y_train,epochs=300, batch_size=16, verbose=1,validation_data=(X_test,y_test))

y_predict = model.predict(X_test) # predictions on test data

# calculate the prediction error
mape = np.sum(abs(y_test[:,0]- y_predict))/np.size(y_test)
print(mape)

y_predict_actual = y_predict*yr+ ym
y_test_actual = y_test[:,0]*yr + ym

plt.plot(y_predict_actual)
plt.plot(y_test_actual)

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.plot(np.arange(57), 1.4*np.ones((57, 1)),'k--',linewidth = 2)
ax.plot(y_predict_actual, color='black',label='Predicted Capacity')
ax.plot(y_test_actual, color='red',label='Actual Capacity')
ax.set(xlabel='Discharge Cycles', ylabel='Capacity(Ah)')
ax.set_xlim([0,57])
ax.legend()