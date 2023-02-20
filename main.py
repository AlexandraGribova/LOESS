import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math

# количество факторов (размерность массива х)
const_n = 2

def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess:
    # методы класса Loess
    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    @staticmethod
    def normalize_array_multidim(array):
        norm_array = np.zeros(len(array[0]))
        # через норму ищем минимальную, максимальную точку
        # ЕСЛИ БУДЕТ БОЛЬШЕ ПАРАМЕТРОВ ТО БУДЕТ И БОЛЬШЕ СЛАГАЕМЫХ МОЖНО СДЕЛАТЬ ЦИКЛ
        for i in range(len(array[0])):
            norm_array[i] = np.sqrt(array[0][i]**2 + array[1][i]**2)
        # ищем в каком индексе лежит мин и макс, так как индекс = номер мин/макс точки
        min_val_index = np.where(norm_array == np.min(norm_array))
        max_val_index = np.where(norm_array == np.max(norm_array))
        new_array = (array - [array[0][min_val_index], array[1][min_val_index]])
        new_array = new_array/([array[0][max_val_index]-array[0][min_val_index],
                                array[1][max_val_index]-array[1][min_val_index]])
        return new_array, \
               [float(array[0][min_val_index]), float(array[1][min_val_index])], \
               [float(array[0][max_val_index]), float(array[1][max_val_index])]

    # конструктор
    # нашли минимальное и максимальное значение, перемасштабировали массив - все его значения между 0 и 1
    def __init__(self, xx, yy, degree=2):
        # масштабируем
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array_multidim(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        # найдем порядковый номер в массиве минимального расстояния
        min_idx = np.argmin(distances)
        # количество элементов в массиве расстояний
        n = len(distances)
        # если работаем с первым или последним эл-ом массива Х
        # вернет равномерно распределенные индексы от 0 до размера окна - 1
        # то есть задаст индексы массива которые принадлежат окну для заданного текущего параметра x'
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        # если текущий параметр Х не находится на краю области
        # min_range - массив который задаст индексы из общего массива Х для окна x'
        min_range = [min_idx]
        # в данном цикле получаем индексы точек с наименьшими расстояниями до x'
        # то есть в окно попадают самые ближние точки-соседи, а не просто 3 справа 3 слева
        while len(min_range) < window:
            # текущее (x') предыдущее и
            i0 = min_range[0]
            #[-1] - последнее значение массива
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (np.array(value) - np.array(self.min_xx)) / (np.array(self.max_xx) - np.array(self.min_xx))

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, degree=2):
        # перемасштабировали данный элемент массива
        n_x = self.normalize_x(x)
        # посчитаем расстояние от текущего элемента х' до всех элементов массива иксов
        # формула расстояния ab=(xa-xb)**2 + (ya - yb)**2
        distances = (self.n_xx[0] - n_x[0])**2 + (self.n_xx[1] - n_x[1])**2
        # задает индексы для окна в заданном массиве для x'
        min_range = self.get_min_range(distances, window)
        #  задает веса для каждого элемента в окне
        weights = self.get_weights(distances, min_range)

        # степень приближающего полинома
        if degree > 1:
            # двумерный массив размерности window - на диагонали 1, вне диагноали 0
            # wm - матрица W
            wm = np.multiply(np.eye(window), weights)
            # массив размерности window*(degree + 1) заполненный единицами
            xm = np.ones((window, degree * const_n + 1))

           # свчитаем матрицу X, которая состоит из перемасшиабированных элементов выборки
           # (каждый столбец - возведение в степень от 0 до степент полниома degree
            for j in range(const_n):
                for i in range(1, degree + 1):
                    xm[:, j*const_n + i] = np.power(self.n_xx[j][min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym

            y = (xm @ np.transpose(beta))[0] #!!!! переставила местами множители
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)


def main():
    # входные данные
    # xx = np.array([[0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
    #                4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
    #                8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
    #                14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
    #                18.7572812],
    #               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                17, 18, 19, 20, 21]])
    # yy = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
    #                213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
    #                227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
    #                160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
    #                243.18828])

    fn = r'C:\Users\sasha\PycharmProjects\OilDiploma\data.csv'
    df = pd.read_csv(fn, sep=';', encoding='cp1251')
    new_df = df.loc[df['ED3'].str.strip() == '205']
    new_df = new_df.sort_values(['ED4', 'ED5'])
    new_df = new_df.drop_duplicates(subset=['ED4', 'ED5'])
    # добыча нефти за каждый месяц последовательно
    new_df['ED14'] = pd.to_numeric(new_df['ED14'].str.replace(',', '.'), errors='coerce')
    new_df['ED15'] = pd.to_numeric(new_df['ED15'].str.replace(',', '.'), errors='coerce')

    # объем добытой нефти
    yy = np.asarray(new_df['ED14'])

    time = np.array(range(1, len(yy) + 1))
    another_param = np.array(new_df['ED5'])
    xx = np.row_stack((time, another_param))
    # дополнительные параметры


    new_df.to_csv('FILE.csv', index=False, sep=";", encoding='cp1251')


    # работа с конкретной выборкой

    # выудили из выборки все данные для одной скважины (выбрала 205)
    # удалили данные для других скважин и повторения в датах для скважины 205
    fn = r'C:\Users\sasha\PycharmProjects\OilDiploma\data.csv'
    df = pd.read_csv(fn, sep=';', encoding='cp1251')
    new_df = df.loc[df['ED3'].str.strip() == '205']
    new_df = new_df.sort_values(['ED4', 'ED5'])
    new_df = new_df.drop_duplicates(subset=['ED4', 'ED5'])
    # добыча нефти за каждый месяц последовательно
    new_df['ED14'] = pd.to_numeric(new_df['ED14'].str.replace(',', '.'), errors='coerce')
    new_df['ED15'] = pd.to_numeric(new_df['ED15'].str.replace(',', '.'), errors='coerce')
    extraction = np.asarray(new_df['ED14'])
    another_param = np.asarray(new_df['ED5'])
    # порядковый номер месяца от начала работы скважины
    time = np.asarray(range(1, len(extraction) + 1))
    new_df.to_csv('FILE.csv', index=False, sep=";", encoding='cp1251')
    # создали экземпляр класса
    loess = Loess(xx, yy)

    # к каждому элементу применили метод
    # указываем степень приближаемого полинома degree
    y_new = np.array([])
    for i in range (len(xx[0])):
            point_of_interest = [xx[0][i], xx[1][i]]
            y = loess.estimate(point_of_interest, window=7, degree=2)
            y_new = np.append(y_new, y)
    plt.plot(xx[0], yy, 'o', label='выборка')
    plt.plot(xx[0], y_new, '-', label='LOESS')
    plt.show()



main()