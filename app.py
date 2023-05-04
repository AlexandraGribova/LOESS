from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from numpy import arange
import matplotlib
import numpy as np
import pandas as pd
import loess

def create_plot(xx, yy, x_new, y_new):
    matplotlib.use('TkAgg')
    w, h = figsize = (5, 3)  # figure size
    fig = matplotlib.pyplot.Figure(figsize=figsize)
    dpi = fig.get_dpi()
    size = (w * dpi, h * dpi)  # canvas size
    t = np.arange(0, 3, .01)
    area = fig.add_subplot(111)
    area.plot(x_new[0], y_new, '-', label='LOESS with param')
    area.plot(xx[0], yy, 'o', label='Выборка')
    area.legend()
    return fig

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def app():
    #-- Меню --#
    menu_def = [["Инструкция"]]
    menu_def_second = [["Инструкция"],
                ["Сохранить",
                             ["Сохранить файл",
                              "Сохранить изображение",
                              "Сохранить файл как...",
                              "Сохранить изображение как..."]]]
    #-- Спинбокс с числом факторов --#
    number = [i for i in range(1, 21)]
    part = [round(i, 2) for i in arange(0.05, 1.05, 0.05)]
    spinbox_param_1 = sg.Spin(number, 1, key='-PARAM_NUM_1-', readonly=True, size=4, enable_events=True)
    spinbox_param_2 = sg.Spin(number, 1, key='-PARAM_NUM_2-', readonly=True, size=4, enable_events=True)
    spinbox_param_3 = sg.Spin(number, 1, key='-PARAM_NUM_3-', readonly=True, size=4, enable_events=True)
    spinbox_param_4 = sg.Spin(number, 1, key='-PARAM_NUM_4-', readonly=True, size=4, enable_events=True)
    spinbox_degree_1 = sg.Spin(number, 2, key='-DEGREE_1-', readonly=True, size=4, enable_events=True)
    spinbox_degree_2 = sg.Spin(number, 2, key='-DEGREE_2-', readonly=True, size=4, enable_events=True)
    spinbox_degree_3 = sg.Spin(number, 2, key='-DEGREE_3-', readonly=True, size=4, enable_events=True)
    spinbox_degree_4 = sg.Spin(number, 2, key='-DEGREE_4-', readonly=True, size=4, enable_events=True)
    spinbox_window_1 = sg.Spin(part, 0.5, key='-WIN_SIZE_1-', readonly=True, size=4, enable_events=True)
    spinbox_window_2 = sg.Spin(part, 0.5, key='-WIN_SIZE_2-', readonly=True, size=4, enable_events=True)
    spinbox_window_3 = sg.Spin(part, 0.5, key='-WIN_SIZE_3-', readonly=True, size=4, enable_events=True)
    spinbox_window_4 = sg.Spin(part, 0.5, key='-WIN_SIZE_4-', readonly=True, size=4, enable_events=True)
    #-- Элементы фрэйма --#
    frame1 = [[sg.Text('Число параметров\t', pad=(15,5)), spinbox_param_1],
             [sg.Text('Степень полинома\t', pad=(15,0)), spinbox_degree_1],
             [sg.Text('Размер окна\t', pad=(15,0)), spinbox_window_1]]
    frame2 = [[sg.Text('Число параметров\t', pad=(15,5)), spinbox_param_2],
              [sg.Text('Степень полинома\t', pad=(15,0)), spinbox_degree_2],
              [sg.Text('Размер окна\t', pad=(15,0)), spinbox_window_2]]
    frame3 = [[sg.Text('Число параметров\t', pad=(15, 5)), spinbox_param_3],
              [sg.Text('Степень полинома\t', pad=(15, 0)), spinbox_degree_3],
              [sg.Text('Размер окна\t', pad=(15, 0)), spinbox_window_3]]
    frame4 = [[sg.Text('Число параметров\t', pad=(15, 5)), spinbox_param_4],
              [sg.Text('Степень полинома\t', pad=(15, 0)), spinbox_degree_4],
              [sg.Text('Размер окна\t', pad=(15, 0)), spinbox_window_4]]
    # -- Главное окно --#
    layout = [[sg.Menubar(menu_def, tearoff=False)],
              [sg.Text('Выберите файл с выборкой')],
              [sg.Text('Файл: '), sg.InputText(), sg.FileBrowse('Выбрать файл')],
              [sg.Frame("", frame1, expand_x=False, key='-FRAME_1'),
               sg.Frame("", frame2, expand_x=False, visible=False, key='-FRAME_2')],
              [sg.Frame("", frame3, expand_x=False, visible=False, key='-FRAME_3'),
               sg.Frame("", frame4, expand_x=False, visible=False, key='-FRAME_4')],
              [sg.Button('Ввести дополнительный параметор', key='-ADD_INPUT-'), sg.Button('Удалить дополнительный параметор', key='-DEL_INPUT-')],
              [sg.Button('Ввод', key='-INPUT-'), sg.Button('Выход', key='-CANCEL-')]]
    window = sg.Window('Имя окна', layout, finalize=True)
    frame = 1
    #-- Цикл для обработки "событий" и получения "значений" входных данных --#
    while True:
        event, values = window.read()
        # -- Если не закрыли окно, то проверяем корректность данных и обрабатываем при нажатии 'Ввод' --#
        if event == sg.WIN_CLOSED or event == '-CANCEL-':
            break
        if event == '-ADD_INPUT-':
            if frame == 1:
                window['-FRAME_2'].update(visible=True)
            if frame == 2:
                window['-FRAME_3'].update(visible=True)
                #window['-ADD_INPUT-'].hide_row()
                #window['-INPUT-'].hide_row()
                window['-FRAME_3'].unhide_row()
                #window['-ADD_INPUT-'].unhide_row()
                #window['-INPUT-'].unhide_row()
            if frame == 3:
                window['-FRAME_4'].update(visible=True)
            frame = frame + 1
        if event == '-DEL_INPUT-':
            if frame == 2:
                window['-FRAME_2'].update(visible=False)
            if frame == 3:
                window['-FRAME_3'].hide_row()
            if frame == 4:
                window['-FRAME_4'].update(visible=False)
            frame = frame - 1
        if event == '-INPUT-':
            if values[1]:
                file = values[1]
                try:
                    f = pd.read_csv(file, sep=';', encoding='cp1251')
                    #-- Заполнение зависимого и независимого вектора --#
                    yy = np.asarray(f['Y'])
                    xx = []
                    for i in range(values['-PARAM_NUM_1-']):
                        name = 'x'+str(i+1)
                        xx.append([])
                        xx[i].extend(np.asarray(f[name]))
                    xx = np.array(xx)
                    #-- Заполнение параметров --#
                    const_n = values['-PARAM_NUM_1-']
                    degree = values['-DEGREE_1-']
                    window_size = values['-WIN_SIZE_1-']
                    # передаваемые параметры:
                    # 1) Многомерный массив хх (каждая строка массива - выборка)
                    # 2) Одномерный массив yy
                    # 3) Степень полинома
                    # 4) Число параметров, от которых зависит вектор уу
                    # 5) Окно (доля выборки которая будет явлться окном)
                    x_new, y_new, eps = loess.MultidimLOESS(xx, yy, degree, const_n, window_size)
                    #-- Вывод результатов --#
                    window.hide()
                    layout_result = [[sg.Menubar(menu_def_second, tearoff=False)],
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
                        if event == "Сохранить файл" or event == "Сохранить файл как...":
                            if event == "Сохранить файл как...":
                                file_name = sg.popup_get_file('Выберите файл или введите название', title="Сохранить файл как...")
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
                        if event == '-OUT_IMG-':
                            plt.savefig('foo.png')



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