from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_svg
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from numpy import arange
import matplotlib
import numpy as np
import pandas as pd
import loess

def informationWindow():
    inf_layout = [[sg.Text('Инструкция')],
                  [sg.Button('Выход', key='-CANCEL_INF-')]]
    window_inf = sg.Window('Инструкция', inf_layout, finalize=True)
    while True:
        event, values = window_inf.read()
        # -- Если не закрыли окно, то проверяем корректность данных и обрабатываем при нажатии 'Ввод' --#
        if event == sg.WIN_CLOSED or event == '-CANCEL_INF-':
            window_inf.close()
            break
def create_plot(xx, yy, x_new, y_new, win_name):
    matplotlib.use('TkAgg')
    w, h = figsize = (7, 5)  # figure size
    fig = matplotlib.pyplot.Figure(figsize=figsize)
    area = fig.add_axes([0.15, 0.25, 0.7, 0.7])
    area.plot(xx[0], yy, 'o')
    for i in range(len(x_new)):
        area.plot(x_new[i], y_new[i], '-', label=win_name[i])
    area.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=4)
    # canvas size
    dpi = fig.get_dpi()
    size = (w * dpi, h * dpi)
    return fig, size

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def app():
    #-- Меню --#
    menu_def = [["Информация", ["Инструкция"]]]
    menu_def_second = [["Информация", ["Инструкция"]],
                ["Сохранить",
                             ["Сохранить файл",
                              "Сохранить изображение",
                              "Сохранить файл как...",
                              "Сохранить изображение как..."]]]
    #-- Спинбокс с числом факторов --#
    number = [i for i in range(1, 21)]
    part = [round(i, 2) for i in arange(0.05, 1.05, 0.05)]

    #-- Самая важная часть, второй раз я это не напишу --#
    def create_row(row_counter):
        row = [sg.pin(
            sg.Col([[sg.Text('Число факторов\t', pad=((15,5), (15,0))),
                     sg.Spin(number, 1, key=('-PARAM_NUM_-', row_counter), pad=(15, (15,0)), readonly=True, size=4, enable_events=True)],
                   [sg.Text('Степень полинома\t', pad=((15,5), 5)),
                    sg.Spin(number, 2, key=('-DEGREE_-', row_counter), pad=(15, 5), readonly=True, size=4, enable_events=True)],
                   [sg.Text('Размер окна\t', pad=((15,5), 0)),
                    sg.Spin(part, 0.5, key=('-WIN_SIZE_-', row_counter), pad=(15, 0), readonly=True, size=4, enable_events=True)],
                   [sg.Button('Удалить', pad=(15,(10,10)), key=('-DEL_INPUT-', row_counter))]],
                   key=('-ROW-', row_counter), element_justification='c'))]
        return row

    # -- Главное окно --#
    layout = [[sg.Menubar(menu_def, tearoff=False)],
              [sg.Text('Выберите файл с выборкой')],
              [sg.Text('Файл: '), sg.InputText(), sg.FileBrowse('Выбрать файл')],
              [sg.Column([create_row(1)], k='-ROW_PANEL1-', vertical_alignment='top'),
               sg.Column([[]], k='-ROW_PANEL2-', vertical_alignment='top')],
              [sg.Button('Ввести дополнительный параметр', key='-ADD_INPUT-')],
              [sg.Button('Ввод', key='-INPUT-'), sg.Button('Выход', key='-CANCEL-')]]
    window = sg.Window('MultiRegression', layout, finalize=True)
    window[('-ROW-',1)].Widget.configure(borderwidth=1, relief=sg.DEFAULT_FRAME_RELIEF)
    frame = 1
    block_number = 1
    active_windows = [1]
    #-- Цикл для обработки "событий" и получения "значений" входных данных --#
    while True:
        event, values = window.read()
        # -- Если не закрыли окно, то проверяем корректность данных и обрабатываем при нажатии 'Ввод' --#
        if event == sg.WIN_CLOSED or event == '-CANCEL-':
            break
        if event == '-ADD_INPUT-':
            if block_number < 4:
                block_number += 1
                frame += 1
                if block_number%2 == 0:
                    window.extend_layout(window['-ROW_PANEL2-'], [create_row(frame)])
                else:
                    window.extend_layout(window['-ROW_PANEL1-'], [create_row(frame)])
                window[('-ROW-', frame)].Widget.configure(borderwidth=1, relief=sg.DEFAULT_FRAME_RELIEF)
                active_windows.append(frame)
        if event[0] == '-DEL_INPUT-':
            if block_number > 1:
                block_number -= 1
                window[('-ROW-', event[1])].update(visible=False)
                active_windows.remove(event[1])
        if event == "Инструкция":
            informationWindow()
        if event == '-INPUT-':
            if values[1]:
                file = values[1]
                try:
                    f = pd.read_csv(file, sep=';', encoding='cp1251')
                    #-- Заполнение зависимого и независимого вектора --#
                    yy = np.asarray(f['Y'])
                    xx_new = []
                    yy_new = []
                    eps_new = []
                    k=0
                    win_name=[]
                    for j in active_windows:
                        xx = []
                        if(values[('-PARAM_NUM_-', j)]):
                            for i in range(values[('-PARAM_NUM_-', j)]):
                                name = 'x'+str(i+1)
                                xx.append([])
                                xx[i].extend(np.asarray(f[name]))
                            xx = np.array(xx)
                            #-- Заполнение параметров --#
                            const_n = values[('-PARAM_NUM_-', j)]
                            degree = values[('-DEGREE_-', j)]
                            window_size = values[('-WIN_SIZE_-', j)]
                            win_name.append('Param number: '+str(values[('-PARAM_NUM_-', j)])+\
                                '\nDegree: '+str(values[('-DEGREE_-', j)])+\
                                '\nWin size: '+str(values[('-WIN_SIZE_-', j)]))
                            # передаваемые параметры:
                            # 1) Многомерный массив хх (каждая строка массива - выборка)
                            # 2) Одномерный массив yy
                            # 3) Степень полинома
                            # 4) Число параметров, от которых зависит вектор уу
                            # 5) Окно (доля выборки которая будет явлться окном)
                            x_new, y_new, eps_crossv, eps_rmse = loess.MultidimLOESS(xx, yy, degree, const_n, window_size)
                            xx_new.append([])
                            yy_new.append([])
                            eps_new.append([])
                            xx_new[k].extend(x_new[0])
                            yy_new[k].extend(y_new)
                            eps_new[k].append(eps_crossv)
                            eps_new[k].append(eps_rmse)
                            k+=1
                    #-- Вывод результатов --#
                    window.hide()
                    data = []
                    for l in range(len(active_windows)):
                        data.append([])
                        data[l].append(win_name[l])
                        data[l].append(eps_new[l][0])
                        data[l].append(eps_new[l][1])
                    headings = ['Заголовок', 'Кросс-валидация', 'RMSE']
                    figure, size_w = create_plot(xx, yy, xx_new, yy_new, win_name)
                    layout_result = [[sg.Menubar(menu_def_second, tearoff=False)],
                                     [sg.Text('График:')],
                                     [sg.Canvas(key='-CANVAS-', size=size_w)],
                                     [sg.Table(values=data, headings=headings, expand_x=True, justification='center',
                                               num_rows=2, key='-TABLE-', row_height=50, tooltip='This is a table',
                                               background_color='white', text_color='black')],
                                     [sg.Button('Выход', key='-CANCEL_RES-')]]
                    window_result = sg.Window('MultiRegression', layout_result, finalize=True)
                    draw_figure(window_result['-CANVAS-'].TKCanvas, figure)
                    #-- Цикл для обработки "событий" и получения "значений" входных данных --#
                    while True:
                        event, values = window_result.read()
                        if event == sg.WIN_CLOSED or event == '-CANCEL_RES-':
                            window_result.close()
                            window.un_hide()
                            break
                        if event == "Инструкция":
                            informationWindow()
                        if event == "Сохранить файл" or event == "Сохранить файл как...":
                            if event == "Сохранить файл как...":
                                file_name = sg.popup_get_file('Выберите файл или введите название файла', title="Сохранить файл как...")
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
                        if event == "Сохранить изображение" or event == "Сохранить изображение как...":
                            if event == "Сохранить изображение как...":
                                img_name = sg.popup_get_file('Выберите директорию или введите название файла', title="Сохранить изображение как...")
                            else:
                                img_name = 'output_img.png'
                            try:
                                figure.savefig(img_name, bbox_inches='tight')
                            except IOError:
                                sg.Popup('Ошибка: файл с именем ' + img_name + ' уже открыт.\n'
                                         'Закройте файл или выберите другую директорию для сохранения.')
                                if event == sg.WIN_CLOSED:
                                    break
                except FileNotFoundError:
                    sg.Popup('Ошибка: файл с именем ' + file + ' не найден')
                    if event == sg.WIN_CLOSED:
                        break
            else:
                sg.Popup('Введите векторы зависимых и независимых переменных')
                if event == sg.WIN_CLOSED:
                    break
    window.close()