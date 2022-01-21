from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras import backend as keras_backend
from keras.utils.np_utils import to_categorical
import numpy as np

random_seed = 42
np.random.seed(random_seed)

# загрузка данных MNIST и сохранение размеров
(X_train, y_train), (X_test, y_test) = mnist.load_data()
image_height = X_train.shape[1]
image_width = X_train.shape[2]
number_of_pixels = image_height * image_width

# конвертирование в плавающую точку
X_train = keras_backend.cast_to_floatx(X_train)
X_test = keras_backend.cast_to_floatx(X_test)

# масштабирование данных в интервал [0, 1]
X_train /= 255.0
X_test /= 255.0

# сохранение оригинальных y_train and y_test
original_y_train = y_train
original_y_test = y_test

# перемещение маркировок данных в версии кодов индивидуальных параметров
number_of_classes = 1 + max(np.append(y_train, y_test))
y_train = to_categorical(y_train, num_classes=number_of_classes)
y_test = to_categorical(y_test, num_classes=number_of_classes)

# переформатирование выборок в решетку 2D, одна строка на изображение
X_train = np.reshape(X_train, [X_train.shape[0], number_of_pixels])
X_test = np.reshape(X_test, [X_test.shape[0], number_of_pixels])


def make_one_hidden_layer_model():
    # создание пустой модели
    model = Sequential()
    # добавление полносвязного скрытого слоя с #узлами = #пикселям
    model.add(Dense(number_of_pixels, activation='relu',
                    input_shape=[number_of_pixels]))
    # добавление выходного слоя с софтмаксом
    model.add(Dense(number_of_classes, activation='softmax'))
    # компиляция модели для превращения ее из спецификации в код
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


one_hidden_layer_model = make_one_hidden_layer_model()  # make the model

one_hidden_layer_history = one_hidden_layer_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test), epochs=20,
    batch_size=256, verbose=2)

one_hidden_layer_model.save('one_hidden_layer_model.h5')
