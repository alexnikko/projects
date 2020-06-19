import numpy as np
import oracles
from scipy.special import expit
from time import time
import utils

class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """

        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        if w_0 is None:
            w_0 = np.random.uniform(size=X.shape[1])

        history = {}
        history['time'] = [0]
        history['func'] = [self.oracle.func(X, y, w_0)]

        for i in range(self.max_iter):
            step = self.step_alpha / ((i + 1) ** self.step_beta)
            start_time = time()
            w_0 -= step * self.oracle.grad(X, y, w_0)
            iter_time = time() - start_time
            history['time'].append(iter_time)
            history['func'].append(self.oracle.func(X, y, w_0))
            if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                break

        self.w = w_0

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """

        return np.sign(X.dot(self.w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """

        prob1 = expit(X.dot(self.w))
        prob2 = 1 - prob1

        return np.array([prob1, prob2])

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """

        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """

        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """

        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic', batch_size=1, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """

        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.oracle = oracles.BinaryLogistic(l2_coef=kwargs['l2_coef'])

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        np.random.seed(self.random_seed)

        if w_0 is None:
            w_0 = np.random.uniform(size=X.shape[1])

        history = {}
        history['epoch_num'] = [0]
        history['time'] = [0]
        history['func'] = [self.oracle.func(X, y, w_0)]
        history['weights_diff'] = []

        self.w = w_0


        processed_objects_num = 0
        objects_num = X.shape[0]
        permutation = np.random.permutation(objects_num)
        i = 0
        indexes = permutation[self.batch_size * i:self.batch_size * (i + 1)]

        k = 1
        step = self.step_alpha / (k ** self.step_beta)

        start_time = time()
        while k < self.max_iter:
            w_0 -= step * self.oracle.grad(X[indexes], y[indexes], w_0)

            processed_objects_num += len(indexes)  # batch_size or <
            epoch_num = processed_objects_num / objects_num

            i += 1
            indexes = permutation[self.batch_size * i:self.batch_size * (i + 1)]

            if epoch_num - history['epoch_num'][-1] >= log_freq:
                history['epoch_num'].append(epoch_num)
                k += 1
                step = self.step_alpha / (k ** self.step_beta)
                square_norm_diff = np.dot(w_0 - self.w, w_0 - self.w)
                history['weights_diff'].append(square_norm_diff)
                iter_time = time() - start_time
                history['time'].append(iter_time)
                history['func'].append(self.oracle.func(X, y, w_0))

                start_time = time()

                if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    break

                self.w = w_0

                permutation = np.random.permutation(objects_num)
                i = 0
                indexes = permutation[self.batch_size * i:self.batch_size * (i + 1)]

        self.w = w_0

        if trace:
            return history
