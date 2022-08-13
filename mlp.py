import numpy as np
import data
import matplotlib.pyplot as plt


# Определим сигмоиду и ее производную по математическому определению;
def sigmoid_act(x, der=False):
    if der:  # производная сигмоиды
        f = 1 / (1 + np.exp(- x)) * (1 - 1 / (1 + np.exp(- x)))
    else:  # функция активации сигмоида
        f = 1 / (1 + np.exp(- x))

    return f


# Определим функцию активации Rectifier Linear Unit (ReLU) - кусочно-линейная функция
def ReLU_act(x, der=False):
    if der:  # производная кусочно-линейной функции это функция Хевисайда - ступенчатая функция
        f = np.heaviside(x, 1)
    else:
        f = np.maximum(x, 0)

    return f


# Номер для каждого слоя перцептрона:
p = 4  # Слой 1
q = 4  # Слой 2

# Устанавливаем параметр скорости обучения
eta = 1 / 623

# 0: На первом шаге обучения случайно подбираем веса модели
w1 = 2 * np.random.rand(p, data.X_train.shape[1]) - 0.5  # Layer 1
b1 = np.random.rand(p)

w2 = 2 * np.random.rand(q, p) - 0.5  # Слой 2
b2 = np.random.rand(q)

wOut = 2 * np.random.rand(q) - 0.5  # Выходной слой
bOut = np.random.rand(1)

mu = []
vec_y = []

# Начинаем проходить по циклу для каждого слоя

for I in range(0, data.X_train.shape[0]):  # цикл:

    # 1: Входные данные
    x = data.X_train[I]

    # 2: Инициализация алгоритма

    # 2.1: Вычисления прямого цикла
    z1 = ReLU_act(np.dot(w1, x) + b1)  # выходной слой 1
    z2 = ReLU_act(np.dot(w2, z1) + b2)  # выходной слой 2
    y = sigmoid_act(np.dot(wOut, z2) + bOut)  # выходной слой 3

    # 2.2: Ошибка вычисления
    delta_Out = (y - data.Y_train[I]) * sigmoid_act(y, der=True)

    # 2.3: Метод обратного распространения ошибки
    delta_2 = delta_Out * wOut * ReLU_act(z2, der=True)  # Второй слой
    delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True)  # Первый слой

    # 3: Градиентная оптимизация
    wOut = wOut - eta * delta_Out * z2
    bOut = bOut - eta * delta_Out

    w2 = w2 - eta * np.kron(delta_2, z1).reshape(q, p)
    b2 = b2 - eta * delta_2

    w1 = w1 - eta * np.kron(delta_1, x).reshape(p, x.shape[0])
    b1 = b1 - eta * delta_1

    # 4. Вычисление функции потерь
    mu.append((1 / 2) * (y - data.Y_train[I]) ** 2)
    vec_y.append(y[0])

# Построим графики
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(0, data.X_train.shape[0]), mu, alpha=0.3, s=4, label='mu')
plt.title('Потери в процессе обучения алгоритма', fontsize=20)
plt.xlabel('Тренировочная выборка', fontsize=16)
plt.ylabel('Лосс', fontsize=16)
plt.show()

# Графики значений функции потерь за 10 итераций обучения
pino = []
for i in range(0, 9):
    pippo = 0
    for m in range(0, 59):
        pippo += vec_y[60 * i + m] / 60
    pino.append(pippo)

plt.figure(figsize=(10, 6))
plt.scatter(np.arange(0, 9), pino, alpha=1, s=10, label='error')
plt.title('Среднее значение функции потерь', fontsize=20)
plt.xlabel('Итерация', fontsize=16)
plt.ylabel('Лосс', fontsize=16)
plt.show()
