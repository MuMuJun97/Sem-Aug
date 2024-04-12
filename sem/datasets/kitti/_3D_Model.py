# -*-coding:utf-8-*-
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def read_model(path):
    obj_model = np.fromfile(path,dtype=np.float32).reshape([-1,3])
    return obj_model

def sphere_coord(pts):
    src_x, src_y, src_z = pts[:,0], pts[:,1], pts[:,2]
    dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
    phi = np.arctan2(src_y / (src_x + 1e-5),(src_x + 1e-5)) # [-pi, pi]
    the = np.arccos(src_z / (dis + 1e-5)) # [0->pi]
    return [dis,phi,the]


if __name__ == '__main__':
    obj_file =  "car.bin"
    obj_model = read_model(obj_file)

    outs = sphere_coord(obj_model)

    # (0,2*pi)-->(0,360)
    phi = (((outs[1] + np.pi) / np.pi)*180).astype(int)

    # (0->180)
    the = ((outs[2]/np.pi) * 180).astype(int)

    dis = outs[0]
    mat = np.zeros((180,360))
    mat[the,phi] = dis
    plt.imshow(mat)
    plt.show()
    print(mat)



