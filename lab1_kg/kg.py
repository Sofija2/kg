import numpy as np
import math
from PIL import Image



def draw_line1(img_mat, x0, y0, x1, y1, color): #пробелма в угадывании количества точек
    step = 1.0/40  #где count - количество точек
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def draw_line2(img_mat, x0, y0, x1, y1, color):  #Небольшой фикс Можно выбирать шаг на основе расстояния между первой и последней точкой
    #проблема в сложный вычислениях
    count = math.sqrt((x0 - x1)**2 + (y0 -y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def draw_line3(img_mat, x0, y0, x1, y1, color):  #Первая проблема: начальная точка можеn оказаться правее конечной. Тогда цикл не
                                                #сработает совсем. Вторая проблема ч: когда шаг по x меньше,
                                                #чем по y. В этом случае появляются разрывы в отрезках.
    

    xchange = False    #Если изменение по x больше, чем изменение по y, поменяем местами x и y
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    
    if (x0 > x1):   #Если начальная точка правее конечной, поменяем их местами (и x, и y, разумеется)
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if xchange:    #
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

def draw_line4(img_mat, x0, y0, x1, y1, color): #Можно выбирать шаг на основе расстояния между первой и последней точкой

    xchange = False    #Если изменение по x больше, чем изменение по y, поменяем местами x и y
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):   #Если начальная точка правее конечной, поменяем их местами (и x, и y, разумеется)
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0 #вычисляем игрик 
    dy = abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if xchange:    
            img_mat[x, y] = color
        else:
           img_mat[y, x] = color

        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def draw_line5(img_mat, x0, y0, x1, y1, color): #МУмножим все части, которые посвящены вычислению шага, на 2*(x1 - x0)

    xchange = False    #Если изменение по x больше, чем изменение по y, поменяем местами x и y
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):   #Если начальная точка правее конечной, поменяем их местами (и x, и y, разумеется)
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*(x1 - x0)*abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if xchange:    
            img_mat[x, y] = color
        else:
           img_mat[y, x] = color

        derror += dy
        if (derror > 2.0*(x1 - x0)*0.5):
            derror -= 2.0*(x1 - x0)*1.0
            y += y_update     
            
def draw_line6(img_mat, x0, y0, x1, y1, color): #Алгоритм Брезенхема

    xchange = False    #Если изменение по x больше, чем изменение по y, поменяем местами x и y
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):   #Если начальная точка правее конечной, поменяем их местами (и x, и y, разумеется)
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        if xchange:    
            img_mat[x, y] = color
        else:
           img_mat[y, x] = color

        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2*(x1 - x0)
            y += y_update
                                

img_mat=np.zeros((200,200,3),dtype=np.uint8)


#img_mat[0:600,0:800]=[255,74,0] # меняет цвет по ргб

#for i in range(600):     #более полная вресия
#    for j in range(800):
#       img_mat[i,j]=[i%256,j%256,(i**2+j)%256]

for i in range (200):
    xo=100
    yo=100
    x1_=100+95*math.cos(i*2*(math.pi)/13)
    x1=int(x1_)
    y1_=100+95*math.sin(i*2*(math.pi)/13)
    y1=int(y1_)
    draw_line5(img_mat, xo, yo, x1, y1, [255,0,0])

img=Image.fromarray(img_mat, mode='RGB')
img.save('img.png')