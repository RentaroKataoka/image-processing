from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os



folder = os.listdir("./cnn/train") #ファイル・ディレクトリの一覧を取得(リスト)
#folder.pop(-1)
image_size = 224
dense_size  = len(folder)

X_train = []
X_val = []
Y_train = []
Y_val = []

for index, name in enumerate(folder): #enumerate：要素のインデックスと要素を同時に取得(ここではindexにインデックス，nameに要素が入る)
    dir = "./cnn/train/" + name
    files = glob.glob(dir + "/*.png") #ファイルを取得
    for i, file in enumerate(files): 
        image = Image.open(file) #画像ファイルの読み込み
        image = image.convert("RGB") #色空間の変更(Lだとグレースケール)
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_train.append(data)
        Y_train.append(index)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.astype('float32') #データ型の変換
X_train = X_train / 255.0

Y_train = np_utils.to_categorical(Y_train, dense_size)


folder = os.listdir("./cnn/val")
image_size = 224
dense_size  = len(folder)

for index, name in enumerate(folder): #enumerate：要素のインデックスと要素を同時に取得(ここではindexにインデックス，nameに要素が入る)
    dir = "./cnn/val/" + name
    files = glob.glob(dir + "/*.png") #ファイルを取得
    for i, file in enumerate(files): 
        image = Image.open(file) #画像ファイルの読み込み
        image = image.convert("RGB") #色空間の変更(Lだとグレースケール)
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_val.append(data)
        Y_val.append(index)

X_val = np.array(X_val)
Y_val = np.array(Y_val)
X_val = X_val.astype('float32') #データ型の変換
X_val = X_val / 255.0
Y_val = np_utils.to_categorical(Y_val, dense_size)


model = Sequential() #層を積み重ねたもの
model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))                                                         #Conv2D(出力フィルタ数, フィルタサイズ, padding='same'or'valid'(ゼロパディングするかどうか), input_shape=画像のサイズ(.shapeで配列の次元の大きさを表す))
model.add(Activation('relu')) #活性化関数ReLUを使用
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #2*2の大きさの最大プーリング層
model.add(Dropout(0.25)) #過学習予防. 全結合の層とのつながりを25%無効化

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #1次元配列に変換
model.add(Dense(512)) #全結合層
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(dense_size))
model.add(Activation('softmax'))

model.summary() #モデルの要約を出力

optimizers ="Adam" #最適化アルゴリズム
results = {}
epochs = 20
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
results[0]= model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs)
model_json_str = model.to_json()
open('mnist_mlp_model.json', 'w').write(model_json_str)
model.save('mnist_mlp_Adam_20_weights.h5');


x = range(epochs)
for k, result in results.items():
    plt.plot(x, result.history['accuracy'], label=k)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

name = 'acc.jpg'
plt.savefig(name, bbox_inches='tight')
plt.close()

for k, result in results.items():
    plt.plot(x, result.history['val_accuracy'], label=k)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

name = 'val_acc.jpg'
plt.savefig(name, bbox_inches='tight')