import numpy as np
import math
from PIL import Image, ImageOps

model = open("model.obj")
texture_img = Image.open("bunny-atlas.jpg")
texture_img = ImageOps.flip(texture_img)

W_T, H_T = texture_img.size
texture_array = np.array(texture_img)

def barac(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img, z_buff, I0, I1, I2, vt0, vt1, vt2):
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
                    u = lambda0 * vt0[0] + lambda1 * vt1[0] + lambda2 * vt2[0]
                    v = lambda0 * vt0[1] + lambda1 * vt1[1] + lambda2 * vt2[1]
                    tex_x = round(W_T * u)
                    tex_y = round(H_T * v)
                    I =  min(max(-1,(lambda0*I0 + lambda1*I1 + lambda2*I2)),0)
                    img[j, i] = -I*texture_array[tex_y, tex_x]
                    
                    #img[j, i] = I
    return img

def project_vertex(vertex, ax, ay, u0, v0, t_z):
    x, y, z = vertex
    z = z + t_z
    return [
        (ax * x / z) + u0,  
        (ay * y / z) + v0, 
        z
    ]

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
    
    trans = []
    for x, y, z in vertices:
        vertex = np.array([x, y, z])
        rotated_vertex = np.dot(R, vertex)  
        x_rot, y_rot, z_rot = rotated_vertex
        
        trans.append([
            x_rot + t_x,
            y_rot + t_y,
            z_rot + t_z
        ])
    
    return trans

V = []
F = []
FT = [] #texture
VT=[]
for line in model:
    s = line.split()
    if s[0] == "v":    
        V.append([float(s[1]), float(s[2]), float(s[3])])
    if s[0] == 'f':
        F.append([int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])])
    if s[0] == 'f':
        FT.append([int(s[1].split('/')[1]), int(s[2].split('/')[1]), int(s[3].split('/')[1])])
    if s[0]=="vt":
        VT.append([float(s[1]), float(s[2])])

#print(VT)



a = math.radians(20) 
b = math.radians(50)  
g = math.radians(10) 

t_x, t_y, t_z = 0.005, -0.04, 0.2

V_T = transform_vertices(V, a, b, g, t_x, t_y, t_z)

vertex_normals = {}

for face in F:

    v0_idx, v1_idx, v2_idx = face[0]-1, face[1]-1, face[2]-1
    

    v0 = np.array(V_T[v0_idx])
    v1 = np.array(V_T[v1_idx])
    v2 = np.array(V_T[v2_idx])
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normal = np.cross(edge1, edge2)
    face_normal = face_normal / np.linalg.norm(face_normal)
    
    for idx in [v0_idx, v1_idx, v2_idx]:
        if idx not in vertex_normals:
            vertex_normals[idx] = []
        vertex_normals[idx].append(face_normal)


for idx in vertex_normals:
    normals = vertex_normals[idx]
    avg_normal = np.mean(normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    vertex_normals[idx] = avg_normal


ax, ay = 5000, 5000
u0, v0 = 1000, 1000

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)  
z_buff = np.full((2000, 2000), np.inf, dtype=np.float32) 

l = np.array([0, 0, 1])

for (face,tex)in zip(F,FT):
    v0_idx, v1_idx, v2_idx = face[0]-1, face[1]-1, face[2]-1
    vt0_idx, vt1_idx, vt2_idx = tex[0]-1, tex[1]-1, tex[2]-1

    v0_3d = V_T[v0_idx]
    v1_3d = V_T[v1_idx]
    v2_3d = V_T[v2_idx]
    
    n0 = vertex_normals[v0_idx]
    n1 = vertex_normals[v1_idx]
    n2 = vertex_normals[v2_idx]
    
    I0 = np.dot(n0, l)
    I1 = np.dot(n1, l)
    I2 = np.dot(n2, l)
    
    v0_proj = project_vertex(v0_3d, ax, ay, u0, v0, t_z)
    v1_proj = project_vertex(v1_3d, ax, ay, u0, v0, t_z)
    v2_proj = project_vertex(v2_3d, ax, ay, u0, v0, t_z)

    vt0= VT[vt0_idx]
    vt1= VT[vt1_idx]
    vt2= VT[vt2_idx]
    
    triangle(
        v0_proj[0], v0_proj[1], v0_proj[2],
        v1_proj[0], v1_proj[1], v1_proj[2],
        v2_proj[0], v2_proj[1], v2_proj[2],
        img_mat, z_buff, I0, I1, I2,
         vt0,vt1,vt2
    )

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')