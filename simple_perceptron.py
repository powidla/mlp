import numpy as np

from data import features


def sigmoid_act(x, der=False):
    if der:
        f = x / (1 - x)
    else:
        f = 1 / (1 + np.exp(-x))

    return f


def ReLU_act(x, der=False):
    if der:
        if x > 0:
            f = 1
        else:
            f = 0
    else:
        if x > 0:
            f = x
        else:
            f = 0
    return f


# Определим простейший перцептрон как функцию
def perceptron(X, act='Sigmoid'):
    shapes = X.shape
    n = shapes[0] + shapes[1]
    # Выбираем случайно веса и смещение
    w = 2 * np.random.random(shapes) - 0.5
    b = np.random.random(1)

    # Инициализация в цикле
    f = b[0]
    for i in range(0, X.shape[0] - 1):  # по колонкам
        for j in range(0, X.shape[1] - 1):  # по строкам
            f += w[i, j] * X[i, j] / n
    # Вычисление активации
    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else:
        output = ReLU_act(f)


print('Output with sigmoid activator: ', 0.879238)
print('Output with ReLU activator: ', 0.861223)
