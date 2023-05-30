import numpy as np
import math


def MultidimLOESS(xx, yy, degree, param_num, part, predict_param):
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
            #return np.log(array), min_val, max_val
            return (array - min_val) / (max_val - min_val), min_val, max_val

        @staticmethod
        def normalize_array_multidim(array):
            norm_array = np.zeros(len(array[0]))
            max_vec = np.empty((param_num, 1))
            min_vec = np.empty((param_num, 1))
            # через норму ищем минимальную, максимальную точку
            for i in range(len(array[0])):
                sum = 0
                for j in range(param_num):
                    sum += array[j][i]**2
                norm_array[i] = np.sqrt(sum)
            # ищем в каком индексе лежит мин и макс, так как индекс = номер мин/макс точки
            min_val_index = np.where(norm_array == np.min(norm_array))
            max_val_index = np.where(norm_array == np.max(norm_array))
            for i in range(param_num):
                min_vec[i] = array[i][min_val_index]
                max_vec[i] = array[i][max_val_index]
            return array, min_vec.flatten(), max_vec.flatten()

        # конструктор
        # нашли минимальное и максимальное значение, перемасштабировали массив - все его значения между 0 и 1
        def __init__(self, xx, yy):
            # масштабируем
            self.n_xx, self.min_xx, self.max_xx = self.normalize_array_multidim(xx)
            self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
            self.degree = degree

        @staticmethod
        def get_min_range(distances, window, crossValidation):
            # найдем порядковый номер в массиве минимального расстояния
            min_idx = np.argmin(distances)
            # количество элементов в массиве расстояний
            n = len(distances)
            # если работаем с первым или последним эл-ом массива Х
            # вернет равномерно распределенные индексы от 0 до размера окна - 1
            # то есть задаст индексы массива которые принадлежат окну для заданного текущего параметра x'
            if min_idx == 0:
                if crossValidation:
                    return np.arange(1, window)
                else:
                    return np.arange(0, window)
            if min_idx == n-1:
                if crossValidation:
                    return np.arange(n - window, n - 1)
                else:
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
            if crossValidation:
                min_range.remove(min_idx)
            return np.array(min_range)

        @staticmethod
        def get_weights(distances, min_range):
            max_distance = np.max(distances[min_range])
            weights = tricubic(distances[min_range] / max_distance)
            return weights

        def normalize_x(self, value):
            return (np.array(value) - np.array(self.min_xx)) / (np.array(self.max_xx) - np.array(self.min_xx))
            #return np.log(value)

        def denormalize_y(self, value):
            return value * (self.max_yy - self.min_yy) + self.min_yy
            #return math.e ** value

        def estimate(self, x, window, degree, crossValidtion, index=0):
            # перемасштабировали данный элемент массива
            #n_x = self.normalize_x(x)
            n_x = x
            n_xx = self.n_xx
            n_yy = self.n_yy
            # посчитаем расстояние от текущего элемента х' до всех элементов массива иксов
            # формула расстояния ab=(xa-xb)**2 + (ya - yb)**2
            if crossValidtion:
                #window = window - 1
                n_xx = np.delete(n_xx, index, 1)
                n_yy = np.delete(n_yy, index)
            distances = 0
            for i in range(param_num):
                distances += (n_xx[i] - n_x[i])**2
            distances = np.sqrt(distances)
            # задает индексы для окна в заданном массиве для x'
            min_range = self.get_min_range(distances, window, 0)
            #  задает веса для каждого элемента в окне
            weights = self.get_weights(distances, min_range)
            #if crossValidtion:
                #window = window - 1
            # двумерный массив размерности window - на диагонали 1, вне диагноали 0
            # wm - матрица W
            wm = np.multiply(np.eye(window), weights)
            # массив размерности window*(degree + 1) заполненный единицами
            xm = np.ones((window, degree * param_num + 1))

           # свчитаем матрицу X, которая состоит из перемасшиабированных элементов выборки
           # (каждый столбец - возведение в степень от 0 до степент полниома degree
            for j in range(param_num):
                for i in range(1, degree + 1):
                    xm[:, j*degree + i] = np.power(n_xx[j][min_range], i)

            ym = n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            self.beta = beta
            xp = [[1]]
            for i in range(len(n_x)):
                for p in range(1, degree + 1):
                    xp.append([math.pow(n_x[i], p)])
            xp = np.array(xp)
            y = (beta @ xp)[0]
            if crossValidtion:
                return y
            else:
                return self.denormalize_y(y)
        def crossValidation(self, point_of_interest, window, degree):
            eps = 0
            y_validation = np.array([])
            for i in range(len(xx[0])):
                point_of_interest = []
                for j in range(param_num):
                    point_of_interest.append(xx[j][i])
                window = int(part * len(yy))
                y = loess.estimate(point_of_interest, window, degree, 1, i)
                y_validation = np.append(y_validation, y)

            for i in range(len(yy)):
                y_normalize = (yy[i] - self.min_yy)/(self.max_yy - self.min_yy)
                eps += (y_normalize - y_validation[i])**2
            return np.sqrt(eps/len(yy))
        def RMSE(self, y_new):
            RMSE = 0
            for i in range(len(y_new)):
                y_normalize = (y_new[i] - self.min_yy)/(self.max_yy - self.min_yy)
                RMSE += (y_normalize - self.n_yy[i]) ** 2
            return np.sqrt(RMSE / len(yy))
        def prediction(self, window, predict_param):
            predict_x = []
            predict_y = []
            #predict_size = int(window * 0.05) ## 5% от размера окна
            for i in range(len(predict_param[0])):
                x_predict = []
                for j in range(len(predict_param)):
                    x_predict.append(predict_param[j][i])
                xp = np.zeros(len(self.beta))
                xp[0] = 1
                for k in range(len(x_predict)):
                    for p in range(1, degree + 1):
                        xp[k*degree+p] = math.pow(x_predict[k], p)
                xp = np.array(xp)
                y_predict = (self.beta @ xp.reshape(param_num*degree + 1, 1))[0]
                predict_x.append(x_predict)
                predict_y.append(self.denormalize_y(y_predict))
            return np.array(predict_x), np.array(predict_y)


    # создали экземпляр класса
    loess = Loess(xx, yy)
    # к каждому элементу применили метод
    # указываем степень приближаемого полинома degree
    y_new = np.array([])
    for i in range(len(xx[0])):
        point_of_interest = []
        for j in range(param_num):
            point_of_interest.append(xx[j][i])
        window = int(part * len(yy))
        y = loess.estimate(point_of_interest, window, degree, 0)
        y_new = np.append(y_new, y)
    # подсчет кросс валидации
    eps_crossv = loess.crossValidation(point_of_interest, window, degree)
    eps_rmse = loess.RMSE(y_new)
    predict_x, predict_y = loess.prediction(window, predict_param)
    return xx, y_new, eps_crossv, eps_rmse, predict_param[0], predict_y