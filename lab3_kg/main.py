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

def project_vertex(vertex, ax, ay, u0, v0, z_offset):
   # """Проективное преобразование 3D -> 2D"""
    x, y, z = vertex
    z = z + z_offset  # Сдвигаем модель, чтобы Z был положительным
    return [
        (ax * x / z) + u0,  # Экранная координата u
        (ay * y / z) + v0,   # Экранная координата v
        z  # Сохраняем z для z-буфера
    ]



def rotate_x(vertex, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = vertex
    y_new = cos_a * y + sin_a * z
    z_new = -sin_a * y + cos_a * z
    return [x, y_new, z_new]

def rotate_y(vertex, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = vertex
    x_new = cos_a * x + sin_a * z
    z_new = -sin_a * x + cos_a * z
    return [x_new, y, z_new]


def rotate_z(vertex, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = vertex
    x_new = cos_a * x - sin_a * y
    y_new = sin_a * x + cos_a * y
    return [x_new, y_new, z]

V = []
F = []
for line in model:
    s = line.split()
    if s[0] == "v":    
        V.append([float(s[1]), float(s[2]), float(s[3])])
    if s[0] == 'f':
        F.append([int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])])

# Центрирование модели
min_coords = np.min(V, axis=0)
max_coords = np.max(V, axis=0)
center = (min_coords + max_coords) / 2
V = [[v[0]-center[0], v[1]-center[1], v[2]-center[2]] for v in V]

# Углы вращения (в радианах)
angle_x = math.radians(20)  # Вращение вокруг X
angle_y = math.radians(150)  # Вращение вокруг Y
angle_z = math.radians(10)  # Вращение вокруг Z

# Применяем вращения ко всем вершинам
V = [rotate_x(v, angle_x) for v in V]
V = [rotate_y(v, angle_y) for v in V]
V = [rotate_z(v, angle_z) for v in V]

# Параметры проекции
ax, ay = 5000, 5000  # Коэффициенты масштабирования
u0, v0 = 1000, 1000  # Центр изображения
z_offset = 0.4  # Сдвиг модели по Z (чтобы вся модель была перед камерой)

# Инициализация изображения и z-буфера
img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)  # Черное изображение 2000x2000
z_buff = np.full((2000, 2000), np.inf, dtype=np.float32)  # Z-буфер

# Отрисовка каждой грани
for face in F:
    # Получаем вершины грани
    v0_idx, v1_idx, v2_idx = face[0]-1, face[1]-1, face[2]-1
    v0_3d = V[v0_idx]
    v1_3d = V[v1_idx]
    v2_3d = V[v2_idx]
    
    # Проецируем вершины на экран с учетом перспективы
    v0_proj = project_vertex(v0_3d, ax, ay, u0, v0, z_offset)
    v1_proj = project_vertex(v1_3d, ax, ay, u0, v0, z_offset)
    v2_proj = project_vertex(v2_3d, ax, ay, u0, v0, z_offset)
    
    # Вычисляем нормаль к грани для определения видимости
    u = np.array(v1_3d) - np.array(v0_3d)
    v = np.array(v2_3d) - np.array(v0_3d)
    n = np.cross(u, v)
    
    l = np.array([0, 0, 1])  # свет

    coss = np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))
    color = (-255 * coss, 0, -100 * coss)
    if coss < 0:
        triangle(
            v0_proj[0], v0_proj[1], v0_proj[2],
            v1_proj[0], v1_proj[1], v1_proj[2],
            v2_proj[0], v2_proj[1], v2_proj[2],
            img_mat, z_buff, color
        )

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
