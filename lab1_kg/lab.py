import numpy as np
import math
from PIL import Image, ImageOps


def draw_line(img_mat, x0, y0, x1, y1, color): #Алгоритм Брезенхема

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

model = open("model_1.obj")
V=[]
F=[]
for line in model:
    s=line.split()
    if s[0] == "v":    
        V.append([float(s[1]),float(s[2]),float(s[3])])
    if s[0]=='f':
        F.append([int(s[1].split('/')[0]),int(s[2].split('/')[0]),int(s[3].split('/')[0])])
    

img_mat=np.zeros((2000,2000,3),dtype=np.uint8)

for i in range(len(F)):
        x0=int(V[F[i][0]-1][0]*10000+1000)
        y0=int(V[F[i][0]-1][1]*10000+500)
        x1=int(V[F[i][1]-1][0]*10000+1000)
        y1=int(V[F[i][1]-1][1]*10000+500)
        x2=int(V[F[i][2]-1][0]*10000+1000)
        y2=int(V[F[i][2]-1][1]*10000+500)
         
        color=(255,255,255)
        print(x0,y0,x1,y1,x2,y2)
        draw_line(img_mat, x0, y0, x1, y1, color)
        draw_line(img_mat, x1, y1, x2, y2, color)
        draw_line(img_mat, x2, y2, x0, y0, color)

img=Image.fromarray(img_mat, mode='RGB')
img=ImageOps.flip(img)
img.save('img1.png')