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

def project_vertex(vertex, ax, ay, u0, v0, t_z):
    x, y, z = vertex
    z = z + t_z  # сдвиг
    return [
        (ax * x / z) + u0,  
        (ay * y / z) + v0, z  ]


def get_rotation_matrix(a, b, g):
  
    R1 = np.array([
        [1, 0, 0],
        [0, math.cos(a), math.sin(a)],
        [0, -math.sin(a), math.cos(a)]
    ])
    
    R2 = np.array([
        [math.cos(b), 0, math.sin(b)],
        [0, 1, 0],
        [-math.sin(b), 0, math.cos(b)]
    ])  

    R3 = np.array([
        [math.cos(g), math.sin(g), 0],
        [-math.sin(g), math.cos(g), 0],
        [0, 0, 1]
    ])
    
    return np.dot(R1, np.dot(R2, R3))

def transform_vertices(vertices, a, b, g, t_x, t_y, t_z):

    R = get_rotation_matrix(a, b, g)
    t = np.array([t_x, t_y, t_z])
    
    transformed = []
    for x, y, z in vertices:
        vertex = np.array([x, y, z])
        rotated_vertex = np.dot(R, vertex)  
        x_rot, y_rot, z_rot = rotated_vertex
        
        transformed.append([
            x_rot + t_x,
            y_rot + t_y,
            z_rot + t_z
        ])
    
    return transformed
V = []
F = []
for line in model:
    s = line.split()
    if s[0] == "v":    
        V.append([float(s[1]), float(s[2]), float(s[3])])
    if s[0] == 'f':
        F.append([int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])])


a = math.radians(20) 
b = math.radians(100)  
g = math.radians(10) 

t_x, t_y, t_z = 0.005, -0.04, 0.2

V_T = transform_vertices(V, a, b, g, t_x, t_y, t_z)

ax, ay = 5000, 5000  # коэф
u0, v0 = 1000, 1000  

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)  
z_buff = np.full((2000, 2000), np.inf, dtype=np.float32) 

for face in F:

    v0_3d = V_T[face[0]-1]
    v1_3d = V_T[face[1]-1]
    v2_3d = V_T[ face[2]-1]
    
    v0_proj = project_vertex(v0_3d, ax, ay, u0, v0, t_z)
    v1_proj = project_vertex(v1_3d, ax, ay, u0, v0, t_z)
    v2_proj = project_vertex(v2_3d, ax, ay, u0, v0, t_z)
    
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
