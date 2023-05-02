from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from numpy import arange
import numpy as np
import pandas as pd
import loess

def create_plot(xx, yy, x_new, y_new):
    plt.plot(xx[0], yy, 'o', label='выборка')
    plt.plot(x_new[0], y_new, '-', label='LOESS with param')
    plt.legend()
    return plt.gcf()

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def app():
    #-- Меню --#
    menu_def = [["Инструкция"],
                ["Сохранить",
                            ["Сохранить в файл",
                                                ["Сохранить", "Сохранить как..."],
                             "Сохранить изображение",
                                                ["Сохранить","Сохранить как..."]]]]
    #-- Спинбокс с числом факторов --#
    number = [i for i in range(1, 21)]
    part = [round(i, 2) for i in arange(0.05, 1.05, 0.05)]
    spinbox_param = sg.Spin(number, 1, key='-PARAM_NUM-', readonly=True, size=4, enable_events=True)
    spinbox_degree = sg.Spin(number, 2, key='-DEGREE-', readonly=True, size=4, enable_events=True)
    spinbox_window = sg.Spin(part, 0.5, key='-WIN_SIZE-', readonly=True, size=4, enable_events=True)
    #-- Элементы фрэйма --#
    frame = [[sg.Text('Число параметров\t\t'), spinbox_param],
             [sg.Text('Степень полинома\t\t'), spinbox_degree],
             [sg.Text('Размер окна\t\t'), spinbox_window]]
    # -- Главное окно --#
    layout = [[sg.Menubar(menu_def, tearoff=False)],
              [sg.Text('Выберите файл с выборкой')],
              [sg.Text('Файл: '), sg.InputText(), sg.FileBrowse('Выбрать файл')],
              [sg.Frame("", frame, expand_x=True), sg.Frame("", [[]], expand_x=True)],
              [sg.Button('Ввести параметры для сравнения', key='-ADD_INPUT-')],
              [sg.Button('Ввод', key='-INPUT-'), sg.Button('Выход', key='-CANCEL-')]]
    window = sg.Window('Имя окна', layout, finalize=True, resizable=True)
    #-- Цикл для обработки "событий" и получения "значений" входных данных --#
    while True:
        event, values = window.read()
        # -- Если не закрыли окно, то проверяем корректность данных и обрабатываем при нажатии 'Ввод' --#
        if event == sg.WIN_CLOSED or event == '-CANCEL-':
            break
        if event == '-INPUT-':
            if values[1]:
                file = values[1]
                try:
                    f = pd.read_csv(file, sep=';', encoding='cp1251')
                    #-- Заполнение зависимого и независимого вектора --#
                    yy = np.asarray(f['Y'])
                    xx = []
                    for i in range(values['-PARAM_NUM-']):
                        name = 'x'+str(i+1)
                        xx.append([])
                        xx[i].extend(np.asarray(f[name]))
                    xx = np.array(xx)
                    #-- Заполнение параметров --#
                    const_n = values['-PARAM_NUM-']
                    degree = values['-DEGREE-']
                    window_size = values['-WIN_SIZE-']
                    # передаваемые параметры:
                    # 1) Многомерный массив хх (каждая строка массива - выборка)
                    # 2) Одномерный массив yy
                    # 3) Степень полинома
                    # 4) Число параметров, от которых зависит вектор уу
                    # 5) Окно (доля выборки которая будет явлться окном)
                    x_new, y_new, eps = loess.MultidimLOESS(xx, yy, degree, const_n, window_size)
                    #-- Вывод результатов --#
                    window.hide()
                    layout_result = [[sg.Menubar(menu_def, tearoff=False)],
                                     [sg.Text('График:')],
                                     [sg.Canvas(key='-CANVAS-')],
                                     [sg.Text('Кроссвалидация: '), sg.Text(eps)],
                                     [sg.Text('Файл: '), sg.InputText('output_file.csv'), sg.FileBrowse('Выбрать файл')],
                                     [sg.Button('Выгрузить данные в файл', key='-OUT_FILE-')],
                                     [sg.Button('Сохранить изображение', key='-OUT_IMG-')],
                                     [sg.Button('Выход', key='-CANCEL_RES-')]]
                    window_result = sg.Window('Window Title', layout_result, finalize=True)
                    draw_figure(window_result['-CANVAS-'].TKCanvas, create_plot(xx, yy, x_new, y_new))
                    #-- Цикл для обработки "событий" и получения "значений" входных данных --#
                    while True:
                        event, values = window_result.read()
                        if event == sg.WIN_CLOSED or event == '-CANCEL_RES-':
                            window_result.close()
                            window.un_hide()
                            break
                        if event == '-OUT_IMG-':
                            plt.savefig('foo.png')
                        if event == '-OUT_FILE-':
                            if values[1]:
                                file_name = values[1]
                            else:
                                file_name = 'output_file.csv'
                            try:
                                d1 = {'Ynew': y_new, 'Y': yy}
                                df = pd.DataFrame(d1)
                                df.to_csv(file_name, sep=';', encoding='cp1251')
                            except IOError:
                                sg.Popup('Ошибка: файл с именем ' + file_name + ' уже открыт.\n'
                                                                                'Закройте файл или выберите другую директорию для сохранения.')
                                if event == sg.WIN_CLOSED:
                                    break


                    # можно вырезать
                    RMSE = 0
                    for i in range(len(y_new)):
                        RMSE += (y_new[i] - yy[i]) ** 2
                    print(np.sqrt(RMSE / len(yy)))
                except FileNotFoundError:
                    sg.Popup('Ошибка: файл с именем ' + file + ' не найден')
                    if event == sg.WIN_CLOSED:
                        break
            else:
                sg.Popup('Введите векторы зависимых и независимых переменных')
                if event == sg.WIN_CLOSED:
                    break
    window.close()