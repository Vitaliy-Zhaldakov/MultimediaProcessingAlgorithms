import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.utils import np_utils
import time

def weight_prune_dense_layer(k_weights, b_weights, k_sparsity):
    # Copy the kernel weights and get ranked indeces of the abs
    kernel_weights = np.copy(k_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(kernel_weights),
            axis=None),
        kernel_weights.shape)

    # Number of indexes to set to 0
    cutoff = int(len(ind[0]) * k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff], ind[1][0:cutoff])
    kernel_weights[sparse_cutoff_inds] = 0.

    # Copy the bias weights and get ranked indeces of the abs
    bias_weights = np.copy(b_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(bias_weights),
            axis=None),
        bias_weights.shape)

    # Number of indexes to set to 0
    cutoff = int(len(ind[0]) * k_sparsity)
    # The indexes in the 1D bias weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff])
    bias_weights[sparse_cutoff_inds] = 0.

    return kernel_weights, bias_weights


def sparsify_model(model, x_test, y_test, k_sparsity):
    # Copying a temporary sparse model from our original
    sparse_model = keras.models.clone_model(model)
    sparse_model.set_weights(model.get_weights())

    # Getting a list of the names of each component (w + b) of each layer
    names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
    # Getting the list of the weights for each component (w + b) of each layer
    weights = sparse_model.get_weights()

    # Initializing list that will contain the new sparse weights
    newWeightList = []

    # Iterate over all but the final 2 layers (the softmax)
    for i in range(0, len(weights) - 1, 2):
        kernel_weights, bias_weights = weight_prune_dense_layer(weights[i],
                                                                weights[i + 1],
                                                                k_sparsity)

        # Append the new weight list with our sparsified kernel weights
        newWeightList.append(kernel_weights)

        # Append the new weight list with our sparsified bias weights
        newWeightList.append(bias_weights)

    # Adding the unchanged weights of the final 2 layers
    for i in range(len(weights), len(weights)):
        unmodified_weight = np.copy(weights[i])
        newWeightList.append(unmodified_weight)

    # Setting the weights of our model to the new ones
    sparse_model.set_weights(newWeightList)

    # Re-compiling the Keras model (necessary for using `evaluate()`)
    sparse_model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy'])

    # Printing the the associated loss & Accuracy for the k% sparsity
    prunTime = time.time()
    score = sparse_model.evaluate(x_test, y_test, verbose=0)
    print('k% weight sparsity: ', k_sparsity,
          '\tTest loss: {:07.5f}'.format(score[0]),
          '\tTest accuracy: {:05.2f} %%'.format(score[1] * 100.),
          "\nВремя работы: ", time.time() - prunTime)

    return sparse_model

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

# print("X_train: " + str(train_X.shape))
# print("Y_train: " + str(train_Y.shape))

# Нормализация тренировочных выборок
x_train = train_X / 255
x_test = test_X / 255
y_train_cat = keras.utils.to_categorical(train_Y, 10)
y_test_cat = keras.utils.to_categorical(test_Y, 10)

# Структура нейронной сети
model = keras.Sequential([Flatten(input_shape=(28, 28, 1)),  Dense(128, activation='relu'), Dense(10, activation='softmax')])
# model = keras.Sequential([Flatten(input_shape=(28, 28, 1)),  Dense(128, activation='sigmoid'), Dense(10, activation='softmax')])
# model = keras.Sequential([Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'), MaxPooling2D((2, 2)),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10,  activation='softmax')])

# Параметры обучения
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение нейронной сети
trainTime = time.time()
history = model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.3)
print("Время обучения: ", time.time() - trainTime, "\n")

# Тестирование нейросети
print("Результат обучения:")
testTime = time.time()
result = model.evaluate(x_test, y_test_cat)
print("Время работы: ", time.time() - testTime)

k_sparcities = [0.0, 0.25, 0.3, 0.4, 0.5]
for i in k_sparcities:
    sparseModel = sparsify_model(model, x_test, y_test_cat, i)
    del sparseModel

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()

n = int(input("Введите номер картинки из тестовой базы MNIST или -1 для выхода из программы: "))
while n != -1:
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print("Вектор выходов: ", res)
    print("Распознана цифра " + str(np.argmax(res)))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
    n = int(input("Введите номер картинки из тестовой базы MNIST или -1 для выхода из программы: "))
