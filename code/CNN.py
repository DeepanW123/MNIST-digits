import tensorflow as tf
from keras import layers, models
import numpy as np
from PIL import Image
import csv


EPOCHS = 20
HEIGHT, WIDTH = 28, 28

def viewImage(image):
    #function to visualize the image from pixel values

    array = np.array(image, dtype=np.uint8)
    array = array.reshape((HEIGHT, WIDTH))

    img = Image.fromarray(array, mode='L')
    img.show()

def initialization(filename):
    #initializing the arrays to store the labels and pixel-values
    label = []
    pixelValues = []

    with open(filename, 'r') as file:
        rows = csv.reader(file)
        if "test" in filename:
            header = next(rows)
        for row in rows:
            label.append(int(row[0]))

            pixels = list(map(int, row[1:]))
            pixelValues.append(pixels)
    
    return label, pixelValues

def makeModel():
    #Training the AI-model

    model = models.Sequential()

    #Convolutional Layer
    model.add(layers.Conv2D(32, (3,3) , input_shape = (28,28,1)))
    model.add(layers.MaxPooling2D((2, 2)))

    #Convolutional Layer
    model.add(layers.Conv2D(64, (3,3)))
    model.add(layers.MaxPooling2D((2, 2)))

    #Flattening the image
    model.add(layers.Flatten())

    #Dropout
    model.add(layers.Dropout(0.4))

    #Hidden layers
    model.add(layers.Dense(activation = "relu", units = 64))
    model.add(layers.Dense(activation = "relu", units = 32))
    model.add(layers.Dense(activation = "relu", units = 32))

    #Dropout
    model.add(layers.Dropout(0.4))

    #Output layer
    model.add(layers.Dense(10, activation="softmax"))

    #Compiling the model
    model.compile(optimizer="adam", loss="categorical_crossentropy" , metrics=['accuracy'])

    return model


def main():

    trainLabel, trainPixelValues = initialization("mnist_train.csv")
    testLabel, testPixelValues = initialization("mnist_test.csv")

    model = makeModel()

    trainPixelValues = np.array(trainPixelValues).reshape(-1,28,28,1)/255.0
    testPixelValues = np.array(testPixelValues).reshape(-1,28,28,1)/255.0

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(trainPixelValues)

    trainLabel =  tf.keras.utils.to_categorical(trainLabel, 10)
    testLabel =  tf.keras.utils.to_categorical(testLabel, 10)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    model.fit(datagen.flow(trainPixelValues, trainLabel, batch_size=32),
            epochs=EPOCHS, verbose=2,
            validation_data=(testPixelValues, testLabel),
            callbacks=[early_stop, checkpoint])

    model.evaluate(testPixelValues, testLabel, verbose=2)

    model.save("trainedModel.h5")

    
main()