from tabnanny import verbose
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os


X_test = []
Y_test = []

folder = os.listdir("./cnn/test")
image_size = 224
dense_size  = len(folder)

for index, name in enumerate(folder): #enumerate：要素のインデックスと要素を同時に取得(ここではindexにインデックス，nameに要素が入る)
    dir = "./cnn/test/" + name
    files = glob.glob(dir + "/*.png") #ファイルを取得
    for i, file in enumerate(files): 
        image = Image.open(file) #画像ファイルの読み込み
        image = image.convert("RGB") #色空間の変更(Lだとグレースケール)
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_test.append(data)
        Y_test.append(index)


X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = X_test.astype('float32') #データ型の変換
X_test = X_test / 255.0
Y_test_backup = Y_test
print(Y_test_backup)
Y_test = np_utils.to_categorical(Y_test, dense_size)


model = load_model("mnist_mlp_Adam_20_weights.h5", )
scores = model.evaluate(X_test, Y_test, verbose=1)
print(scores)

predict_prob = model.predict(X_test)
predict_classes=np.argmax(predict_prob, axis=1)
# true_classes = np.argmax(Y_test, 1)
print(confusion_matrix(Y_test_backup, predict_classes))