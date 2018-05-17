from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

dataset = np.loadtxt("pima.csv", delimiter=",")

x = dataset[:, 0:8]
y = dataset[:,8]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,epochs=150, batch_size=10)

scores = model.evaluate(x_test, y_test)


pred = model.predict(x_test)

for i in range(len(pred)):
    pred[i] = np.round(pred[i])
    
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)