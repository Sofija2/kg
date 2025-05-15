import numpy as np
import math
from PIL import Image, ImageOps
import time

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 100
AX = 5000  
AY = 5000 
U0 = IMAGE_WIDTH // 2  
V0 = IMAGE_HEIGHT // 2  

def render_model(model_path, texture_path=None, 
                 a=0, b=0, g=0,  t_x=0.005,  t_y = -0.04, t_z =  0.2,
                 scale_x=1.0, scale_y=1.0, scale_z=1.0,
                 shift_x=0.0, shift_y=0.0, shift_z=0.0,
                 use_quaternion=False, q=None,
                 img_mat=None, z_buff=None):
    

    if img_mat is None:
        img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    if z_buff is None:
        z_buff = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.inf, dtype=np.float32)
    

    model = open(model_path)
    
    if texture_path:
        try:
            texture_img = Image.open(texture_path)
            texture_img = ImageOps.flip(texture_img)
            W_T, H_T = texture_img.size
            texture_array = np.array(texture_img)
            has_texture = True
        except:
            print(f"Ошибка загрузки текстуры {texture_path}, рендеринг без текстуры")
            has_texture = False
    else:
        has_texture = False
    
    V = []
    F = []
    FT = []
    VT = []
    
    for line in model:
        s = line.strip().split()
        if not s:
            continue
        if s[0] == 'v':
            x = float(s[1]) * scale_x + shift_x
            y = float(s[2]) * scale_y + shift_y
            z = float(s[3]) * scale_z + shift_z
            V.append([x, y, z])
        elif s[0] == 'vt':
            VT.append([float(s[1]), float(s[2])])
        elif s[0] == 'f':
            f = [tuple(part.split('/')) for part in s[1:]]
            vertex_indices = [int(v[0]) for v in f]
            texture_indices = [int(v[1]) for v in f if len(v) > 1]
            # 4
            for i in range(1, len(vertex_indices) - 1):
                F.append([
                    vertex_indices[0],
                    vertex_indices[i],
                    vertex_indices[i + 1]
                ])
                if texture_indices:
                    FT.append([
                        texture_indices[0],
                        texture_indices[i],
                        texture_indices[i + 1]
                    ])
    
   
    if use_quaternion and q is not None:
        V_T = transform_vertices_quat(V, q, t_x, t_y, t_z)
    else:
        V_T = transform_vertices(V, math.radians(a), math.radians(b), math.radians(g), t_x, t_y, t_z) # трансформа все
    
    
    vertex_normals = {}
    for face in F:
        v0_idx, v1_idx, v2_idx = face[0]-1, face[1]-1, face[2]-1
        v0 = np.array(V_T[v0_idx])
        v1 = np.array(V_T[v1_idx])
        v2 = np.array(V_T[v2_idx])
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm != 0:
            face_normal /= norm
        for idx in [v0_idx, v1_idx, v2_idx]:
            if idx not in vertex_normals:
                vertex_normals[idx] = []
            vertex_normals[idx].append(face_normal)
    
    for idx in vertex_normals:
        normals = vertex_normals[idx]
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm != 0:
            avg_normal /= norm
        vertex_normals[idx] = avg_normal
    
    # Вектор освещения
    l = np.array([0, 0, 1])
    
    # Рендеринг треугольников
    for idx, face in enumerate(F):
        v0_idx, v1_idx, v2_idx = face[0]-1, face[1]-1, face[2]-1
        
        # Получаем текстурные координаты 
        if idx < len(FT) and FT and VT:
            vt0_idx, vt1_idx, vt2_idx = FT[idx][0]-1, FT[idx][1]-1, FT[idx][2]-1
            vt0 = VT[vt0_idx]
            vt1 = VT[vt1_idx]
            vt2 = VT[vt2_idx]
        else:
            # Если нет текстурных координат, используем нулевые
            vt0 = vt1 = vt2 = [0, 0]
        
        v0_3d = V_T[v0_idx]
        v1_3d = V_T[v1_idx]
        v2_3d = V_T[v2_idx]
        
        n0 = vertex_normals[v0_idx]
        n1 = vertex_normals[v1_idx]
        n2 = vertex_normals[v2_idx]
        
        I0 = np.dot(n0, l)
        I1 = np.dot(n1, l)
        I2 = np.dot(n2, l)
        
        v0_proj = project_vertex(v0_3d, AX, AY, U0, V0, t_z)
        v1_proj = project_vertex(v1_3d, AX, AY, U0, V0, t_z)
        v2_proj = project_vertex(v2_3d, AX, AY, U0, V0, t_z)
        
        triangle(
            v0_proj[0], v0_proj[1], v0_proj[2],
            v1_proj[0], v1_proj[1], v1_proj[2],
            v2_proj[0], v2_proj[1], v2_proj[2],
            img_mat, z_buff, I0, I1, I2,
            vt0, vt1, vt2,
            has_texture=has_texture,
            texture_array=texture_array if has_texture else None,
            W_T=W_T if has_texture else 1,
            H_T=H_T if has_texture else 1
        )
    
    return img_mat, z_buff

def triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, img, z_buff, I0, I1, I2, vt0, vt1, vt2,
             has_texture=False, texture_array=None, W_T=1, H_T=1):  # in tri
    # Определяем границы треугольника с учетом границ изображения
    xmin = int(max(0, min(x0, x1, x2)))
    ymin = int(max(0, min(y0, y1, y2)))
    xmax = int(min(IMAGE_WIDTH-1, max(x0, x1, x2) + 1))
    ymax = int(min(IMAGE_HEIGHT-1, max(y0, y1, y2) + 1))
    
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambda0, lambda1, lambda2 = barac(i, j, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0*z0 + lambda1*z1 + lambda2*z2
                if z < z_buff[j, i]:
                    z_buff[j, i] = z
                    I = min(max(-1, (lambda0*I0 + lambda1*I1 + lambda2*I2)), 0)
                    
                    if has_texture and texture_array is not None:
                        u = lambda0*vt0[0] + lambda1*vt1[0] + lambda2*vt2[0]
                        v = lambda0*vt0[1] + lambda1*vt1[1] + lambda2*vt2[1]
                        tex_x = min(max(0, round(W_T * u)), W_T-1)
                        tex_y = min(max(0, round(H_T * v)), H_T-1)
                        color = -I * texture_array[tex_y, tex_x]
                    else:
                        # Если нет текстуры используем серый цвет 
                        gray = int(-I * 255)
                        color = np.array([gray, gray, gray])
                    
                    img[j, i] = color
    return img

def barac(x, y, x0, y0, x1, y1, x2, y2):
    den = ((x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2))
    if den == 0:
        return 0, 0, 0
    lambda0 = ((x - x2)*(y1 - y2) - (x1 - x2)*(y - y2)) / den
    lambda1 = ((x0 - x2)*(y - y2) - (x - x2)*(y0 - y2)) / den
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def transform_vertices_quat(vertices, q, t_x, t_y, t_z):
    R = quaternion_to_matrix(q)
    trans = []
    for x, y, z in vertices:
        vertex = np.array([x, y, z])
        rotated_vertex = np.dot(R, vertex)
        trans.append([
            rotated_vertex[0] + t_x,
            rotated_vertex[1] + t_y,
            rotated_vertex[2] + t_z
        ])
    return trans

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


def quae_add(normal1, deg_1):
    deg = math.radians(deg_1)
    sin_half = math.sin(deg / 2)
    cos_half = math.cos(deg / 2)
    nx, ny, nz = normal1
    norm = math.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    return (cos_half, nx * sin_half, ny * sin_half, nz * sin_half)


def quaternion_to_matrix(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

def save_image(img_mat, filename):
    img = Image.fromarray(img_mat, mode='RGB')
    img = ImageOps.flip(img)
    img.save(filename)

if __name__ == "__main__":
    start_time = time.time()
    print("Начало обработки...")
    normal1 = (0, 1, 0)
    gard = 45
    
    
    print("Рендеринг  модели 1...")
    img_mat, z_buff = render_model(
        model_path="mod.obj",
        texture_path="bunny-atlas.jpg",
        use_quaternion=True,
        q=quae_add(normal1,gard), 
        #scale_x=1.5, scale_y=1.5, scale_z=1.5,  
        shift_x=-0.1,  
    )
    
    print("Рендеринг  модели 2...")
    img_mat, z_buff = render_model(
        model_path="mod.obj",
        #texture_path="bunny-atlas.jpg",
        #scale_x=1.5, scale_y=1.5, scale_z=1.5,  
        shift_x=0.1,    
        img_mat=img_mat,
        z_buff=z_buff
    )

    # Сохранение результата
    save_image(img_mat, 'result2.png')
    
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")