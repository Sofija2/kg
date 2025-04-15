import numpy as np
import math
import random
from PIL import Image, ImageOps

model = open("model.obj")

def barac(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img, z_buff, color):
    xmin = int(max(0, min(x0, x1, x2)))
    ymin = int(max(0, min(y0, y1, y2)))
    xmax = int(min(1999, max(x0, x1, x2) + 1))
    ymax = int(min(1999, max(y0, y1, y2) + 1))

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barac(i, j, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buff[j, i]:
                    z_buff[j, i] = z
                    img[j, i] = color
    return img

def rotate_y(vertex, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = vertex
    x_new = cos_a * x + sin_a * z
    z_new = -sin_a * x + cos_a * z
    return [x_new, y, z_new]

V = []
F = []
for line in model:
    s = line.split()
    if s[0] == "v":    
        V.append([float(s[1]), float(s[2]), float(s[3])])
    if s[0] == 'f':
        F.append([int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])])

angle_y = math.radians(30)  # Угол поворота вокруг оси Y (30 градусов)
V = [rotate_y(v, angle_y) for v in V]

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
z_buff = np.full((2000, 2000), np.inf, dtype=np.float32)

u0,v0=1000,1000
ax=100
ay=100

for i in range(len(F)):
    x0 = (V[F[i][0] - 1][0]) * 10000 + 1000
    y0 = (V[F[i][0] - 1][1]) * 10000 + 500
    x1 = (V[F[i][1] - 1][0]) * 10000 + 1000
    y1 = (V[F[i][1] - 1][1]) * 10000 + 500
    x2 = (V[F[i][2] - 1][0]) * 10000 + 1000
    y2 = (V[F[i][2] - 1][1]) * 10000 + 500
    z0 = (V[F[i][0] - 1][2]) * 10000 + 1000   
    z1 = (V[F[i][1] - 1][2]) * 10000 + 1000
    z2 = (V[F[i][2] - 1][2]) * 10000 + 1000

    u = np.array([x1 - x0, y1 - y0, z1 - z0])
    v = np.array([x2 - x0, y2 - y0, z2 - z0])

    n = np.cross(u, v)
    
    l = np.array([0, 0, 1])  # свет

    coss = np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))
    color = (-255 * coss, 0, -100 * coss)
    if coss < 0:
        triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat, z_buff, color)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
