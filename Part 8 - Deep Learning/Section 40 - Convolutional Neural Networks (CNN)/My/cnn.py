# Convolutional Neural Network

# PART 1 Building CNN

# Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np

# Initialising CNN
classifier = Sequential()

# Step 1 Covolutional Layer
classifier.add(Convolution2D(32, (3, 3) , input_shape = (64, 64, 3), activation = 'relu' ))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Another convolutional and maxPooling layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening 
classifier.add(Flatten())

#  step 4 - Full Connection(Hidden layer)
classifier.add(Dense(units = 128, activation = 'relu'))

# Output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training = train_datagen.flow_from_directory('dataset/training_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

test = test_datagen.flow_from_directory('dataset/test_set',
                                        target_size=(64, 64),
                                        batch_size=32,
                                        class_mode='binary')

#traning = 'dataset/training_set'
#test = 'dataset/test_set'
#classifier.fit(np.array('dataset/training_set'), np.array('dataset/test_set'), np_epoch = 20, validation_data=())
#classifier.fit(np.array(traning), np.array(test), epochs = 20)

classifier.fit_generator(training,
                         samples_per_epoch = 4000,
                         nb_epoch = 25,
                         validation_data = test,
                         nb_val_samples = 200)

from keras.preprocessing import image
img = image.load_img('dataset/dog.jpg', target_size=(64,64))

test = test_datagen.flow_from_directory('dataset/test_set',
                                        target_size=(64, 64),
                                        batch_size=32,
                                        class_mode='binary')

img, lab = next(test)

result = classifier.predict_generator(test, steps = 1, verbose = 0)

result = np.round(result)

import matplotlib.pyplot as plt
plt.plot(img, lab)

acc = next(test)[1]

from sklearn.metrics import accuracy_score
print(accuracy_score(acc, result))

classifier.predict_classes(test)

model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save("classifier.h5")
classifier.save_weights("classify.h5")

import tensorflow as tf
from keras.models import load_model, Model
from keras import backend as K

sess = tf.Session()

model = load_model('classifier.h5')

model.load_weights('classify.h5')

from keras.preprocessing import image
img = image.load_img('dataset/dog.jpg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)



gd = sess.graph.as_graph_def()
gd.node[:2]
x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
y = model(x)

import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

model_output = classifier.output

#Predict Class of image
from keras.preprocessing import image
img = image.load_img('dataset/cat.jpg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
print(img)

pred = []

for i in test:
    img = image.load_img(i, target_size=(64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    #pred.append(model.predict(img))

d = model.predict(img)

result = np.array_str(np.argmax(model.predict(img)))


orig_scores = sess.run(y, feed_dict={x: img, K.learning_phase(): False})


#result = classifier.predict_classes(img)
result = classifier.predict_on_batch(img)

#img = image.load_img('dataset/cat.jpg', target_size=(64,64))
#img = image.img_to_array(img)
#img = np.expand_dims(img, axis=0)
#result2 = classifier.predict_on_batch(img)