import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import matplotlib.pyplot as plt
from keras.models import model_from_json
num_classes =10
input_shape =(28,28,1)
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

# scale images to the [0,1] i.e normalize the image
x_train= x_train.astype('float32')/255
x_test= x_test.astype('float32')/255
x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)
# convert class vectors to binary class matrices
y_train= keras.utils.to_categorical(y_train,num_classes)
y_test= keras.utils.to_categorical(y_test,num_classes)
x=x_test[4,:,:,:]
print(x_train.shape)
print(x_test.shape)
#print(x.shape)

json_file = open('model.json', 'r')
    
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) 
    # load weights into new model
loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
   #dim=(28,28)
    #print(curr_num.shape)
    #x= cv2.resize(curr_num,dim,interpolation=cv2.INTER_AREA)
   # print(x.shape)

x=cv.imread('re7.jpg',cv.IMREAD_GRAYSCALE)
x=np.expand_dims(x,2)
x=np.expand_dims(x,0)
y=np.argmax(loaded_model.predict(x), axis=-1)
print(y)
# Build model
'''
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
batch_size = 128
epochs = 4

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('moodel accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper left')
plt.show()
x=np.expand_dims(x,0)
y=np.argmax(model.predict(x), axis=-1)
z="The predicted number is "+str(y[0])
print(z)
cv.waitKey(0)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''