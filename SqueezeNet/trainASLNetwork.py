import numpy as np
import tensorflow
import datetime
# Load the TensorBoard notebook extension
import tensorboard
import glob
import os
import cv2
from tensorflow.keras.layers import Conv2D, Add, Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
train_dir = "./asl_alphabet_train/asl_alphabet_train"
eval_dir = "./asl_alphabet_test/asl_alphabet_test"
batch_size = 32
imageSize = 64
target_dims = (64, 64, 3)
num_classes = 29
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#Helper function to load images from given directories
def load_images(directory, uniq_labels, evalData=False):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        if evalData == False:
            #i=0
            for file in os.listdir(directory + "/" + label):
                #if i>10:
                #    break
                filepath = directory + "/" + label + "/" + file
                image = cv2.resize(cv2.imread(filepath), (64, 64))
                images.append(image)
                labels.append(idx)
                #i=i+1
        else:
            for file in os.listdir(directory):
                filepath = directory + "/" + file
                image = cv2.resize(cv2.imread(filepath), (64, 64))
                images.append(image)
                labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)


def trainModel():
    uniq_labels = sorted(os.listdir(train_dir))
    X_eval, y_eval = None, None
    print("loading evaluation images")
    X_eval, y_eval = load_images(eval_dir, uniq_labels, evalData=True)
    print("loading train/test images")
    images, labels = load_images(train_dir, uniq_labels)

    print("done loading images")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(images,
                                                        labels,
                                                        test_size=0.1,
                                                        stratify=labels)

    n = len(uniq_labels)
    train_n = len(X_train)
    test_n = len(X_test)

    print("Total number of symbols: ", n)
    print("Number of training images: ", train_n)
    print("Number of testing images: ", test_n)

    eval_n = len(X_eval)
    print("Number of evaluation images: ", eval_n)

    y_train_in = y_train.argsort()
    y_train = y_train[y_train_in]
    X_train = X_train[y_train_in]
    y_test_in = y_test.argsort()
    y_test = y_test[y_test_in]
    X_test = X_test[y_test_in]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_eval = to_categorical(y_eval)
    print(y_train[0])
    print(len(y_train[0]))
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_eval = X_eval.astype('float32') / 255.0

    inputs = Input(shape=target_dims)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
    net = LeakyReLU()(net)

    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
    net = LeakyReLU()(net)
    net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
    net = LeakyReLU()(net)

    shortcut = net

    net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)
    net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)

    net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)
    net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
    net = BatchNormalization(axis=3)(net)
    net = LeakyReLU()(net)
    net = Add()([net, shortcut])

    net = GlobalAveragePooling2D()(net)
    net = Dropout(0.2)(net)

    net = Dense(128, activation='relu')(net)
    outputs = Dense(num_classes, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    hist = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, callbacks=[tensorboard_callback])
    model.save('aslClassifier.h5')
    return model


def main():
    print("now loading images")
    #check if saved model exists
    foundModel = glob.glob("*.h5")
    if len(foundModel) != 0:
        model = load_model(foundModel[0])
        model.summary()
        """ 
        score = model.evaluate(x=X_test, y=y_test, verbose=0)
        print('Accuracy for test images:', round(score[1] * 100, 3), '%')
        score = model.evaluate(x=X_eval, y=y_eval, verbose=0)
        print('Accuracy for evaluation images:', round(score[1] * 100, 3), '%')

        y_eval_pred = model.predict(X_eval, batch_size=64, verbose=0)
        print(y_eval_pred) """
    else:
        model = trainModel()
        model.summary()
    # now that you have loaded the network, predict a test sample
    for file in os.listdir(eval_dir):
        filepath = eval_dir + "/" + file
        test_image = cv2.resize(cv2.imread(filepath), (64, 64))
        test_image = test_image.astype('float32') / 255.0
        print(file, test_image.shape)
        print(np.argmax(model.predict(np.expand_dims(test_image, axis=0))))

if __name__ == '__main__':
    main()
