import numpy as np
import math
import mlp
from data import X_test, Y_train, X_train

'''
Многослойной перцептрон
'''


class MLP:
    """
    Инициализация класса перцептрона;
    w, b, phi = векторы весов, смещения и пустой массив
    mu = функция потерь
    eta = параметр обучения. Можно усовершенствовать методом set
    """

    def __init__(self):
        self.HiddenLayer = []
        self.w = []
        self.b = []
        self.phi = []
        self.mu = []
        self.eta = 1  # set up the proper Learning Rate!!

    def add(self, lay=(4, 'ReLU')):
        self.HiddenLayer.append(lay)

    '''
    Метод прямого вычисления весов 
    '''

    @staticmethod
    def FeedForward(w, b, phi, x):
        return phi(np.dot(w, x) + b)

    '''
    Алгоритм обратного распространения ошибки для вычисления градиентов и весов
    '''

    def BackPropagation(self, x, z, Y, w, b, phi):
        self.delta = []

        # Инициализируем веса и байес
        self.W = []
        self.B = []

        # Вычисление ошибки
        self.delta.append((z[len(z) - 1] - Y) * phi[len(z) - 1](z[len(z) - 1], der=True))

        '''back-prop'''
        # Итерируемся по объектам
        for i in range(0, len(z) - 1):
            self.delta.append(
                np.dot(self.delta[i], w[len(z) - 1 - i]) * phi[len(z) - 2 - i](z[len(z) - 2 - i], der=True))

        self.delta = np.flip(self.delta, 0)

        # Нормализация на размерность объектов
        self.delta = self.delta / self.X.shape[0]

        '''Градиентная оптимизация'''
        # Начинаем с первого слоя, который связан с входным слоем
        self.W.append(w[0] - self.eta * np.kron(self.delta[0], x).reshape(len(z[0]), x.shape[0]))
        self.B.append(b[0] - self.eta * self.delta[0])

        # Скрытые слои
        for i in range(1, len(z)):
            self.W.append(w[i] - self.eta * np.kron(self.delta[i], z[i - 1]).reshape(len(z[i]), len(z[i - 1])))
            self.B.append(b[i] - self.eta * self.delta[i])

        # Возвращаем веса и баейс
        return np.array(self.W), np.array(self.B)

    '''
    Процесс фитинга для модели:вычисление градиентов, сброс весов и снова вычисление
    '''

    def Fit(self, X_train, Y_train):
        print('Start fitting...')
        '''
        Входной слой
        '''
        self.X = X_train
        self.Y = Y_train

        '''
        Добавляем в инициализацию скрытые слои 
        '''
        print('Model recap: \n')
        print('You are fitting an MLP with the following amount of layers: ', len(self.HiddenLayer))

        for i in range(0, len(self.HiddenLayer)):
            print('Layer ', i + 1)
            print('Number of neurons: ', self.HiddenLayer[i][0])
            if i == 0:
                # Ссылка на статью с реализацией ArXiv:1502.01852
                self.w.append(np.random.randn(self.HiddenLayer[i][0], self.X.shape[1]) / np.sqrt(2 / self.X.shape[1]))
                self.b.append(np.random.randn(self.HiddenLayer[i][0]) / np.sqrt(2 / self.X.shape[1]))

                # Функция активации
                for act in Activation_function.list_act():
                    if self.HiddenLayer[i][1] == act:
                        self.phi.append(Activation_function.get_act(act))
                        print('\tActivation: ', act)

            else:
                # ArXiv:1502.01852
                self.w.append(np.random.randn(self.HiddenLayer[i][0], self.HiddenLayer[i - 1][0]) / np.sqrt(
                    2 / self.HiddenLayer[i - 1][0]))
                self.b.append(np.random.randn(self.HiddenLayer[i][0]) / np.sqrt(2 / self.HiddenLayer[i - 1][0]))

                for act in Activation_function.list_act():
                    if self.HiddenLayer[i][1] == act:
                        self.phi.append(Activation_function.get_act(act))
                        print('\tActivation: ', act)

        '''
        Обучение модели
        '''
        for I in range(0, self.X.shape[0]):  # loop over the training set
            '''
            Прямой порядок вычисления
            '''
            self.z = []

            self.z.append(self.FeedForward(self.w[0], self.b[0], self.phi[0], self.X[I]))  # First layers

            for i in range(1, len(self.HiddenLayer)):
                self.z.append(self.FeedForward(self.w[i], self.b[i], self.phi[i], self.z[i - 1]))

            '''
            Метод обратного распространения ошибки
            '''
            self.w, self.b = self.BackPropagation(self.X[I], self.z, self.Y[I], self.w, self.b, self.phi)

            '''
            Целевая функция
            '''
            self.mu.append(
                (1 / 2) * np.dot(self.z[len(self.z) - 1] - self.Y[I], self.z[len(self.z) - 1] - self.Y[I])
            )

        print('Fit done. \n')

    '''
    Предсказания модели
    '''

    def predict(self, X_test):

        print('Starting predictions...')

        self.pred = []
        self.XX = X_test

        for I in range(0, self.XX.shape[0]):

            '''
            Прямой ход + обратный
            '''
            self.z = []

            self.z.append(self.FeedForward(self.w[0], self.b[0], self.phi[0], self.XX[I]))  # First layer

            for i in range(1, len(self.HiddenLayer)):  # loop over the layers
                self.z.append(self.FeedForward(self.w[i], self.b[i], self.phi[i], self.z[i - 1]))

            # Append the prediction;
            # We now need a binary classifier; we this apply an Heaviside Theta and we set to 0.5 the threshold
            # if y < 0.5 the output is zero, otherwise is zero
            self.pred.append(
                np.heaviside(self.z[-1] - 0.5, 1)[0])  # NB: self.z[-1]  is the last element of the self.z list

        print('Predictions done. \n')

        return np.array(self.pred)

    '''
    Метод метрик оценки качества
    '''

    def get_accuracy(self):
        return np.array(self.mu)

    def get_avg_accuracy(self):
        self.batch_loss = []
        for i in range(0, 10):
            self.loss_avg = 0
            # math.ceil method
            # int(math.ceil((self.X.shape[0]-10) / 10.0))    - 1
            for m in range(0, (int(math.ceil((self.X.shape[0] - 10) / 10.0))) - 1):
                # self.loss_avg += self.mu[60*i+m]/60
                self.loss_avg += self.mu[(int(math.ceil((self.X.shape[0] - 10) / 10.0))) * i + m] / (
                    int(math.ceil((self.X.shape[0] - 10) / 10.0)))
            self.batch_loss.append(self.loss_avg)
        return np.array(self.batch_loss)

    '''
    Устанавливаем скорость обучения
    '''

    def set_learning_rate(self, et=1):
        self.eta = et


'''
Создание класса слой
'''


class layers:
    """
    Layer метод: стандартный вызов.

    """

    def layer(p=4, activation='ReLU'):
        return (p, activation)


'''
Класс функций активации
'''


class Activation_function(MLP):

    def __init__(self):
        super().__init__()

    '''
    Определим сигмоиду и ее производную
    '''

    def sigmoid_act(x, der=False):
        if der:  # производная сигмоиды
            f = 1 / (1 + np.exp(- x)) * (1 - 1 / (1 + np.exp(- x)))
        else:  # сигмоида
            f = 1 / (1 + np.exp(- x))
        return f

    '''
    ReLU
    '''

    def ReLU_act(x, der=False):
        if der:  # ступенчатая функция
            f = np.heaviside(x, 1)
        else:
            f = np.maximum(x, 0)
        return f

    @staticmethod
    def list_act():
        return ['sigmoid', 'ReLU']

    def get_act(string='ReLU'):
        if string == 'ReLU':
            return mlp.ReLU_act
        elif string == 'sigmoid':
            return mlp.sigmoid_act
        else:
            return mlp.sigmoid_act
