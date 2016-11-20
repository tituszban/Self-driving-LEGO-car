import numpy as np
import cv2
import urllib
import time
import os.path
import clr
import math as m
from PyMutexController import MutexController

def line_in_config(cnf, name):
    for l in cnf:
        spt = l.split('>')
        if spt[0] == name:
            return spt[1].replace('\n', '')

def str_2_tuple(s):
    sp = s.split(',')
    return (int(sp[0]), int(sp[1]))

config = []

f = open('logger.config', 'r')
for l in f:
    config.append(l)
f.close()


dispResolution = str_2_tuple(line_in_config(config, 'disp_Resolution'))
saveResolution = str_2_tuple(line_in_config(config, 'save_Resolution'))
data_path = line_in_config(config, 'data_path')
url = line_in_config(config, 'url')
mutex_name = line_in_config(config, 'mutex_name')
timerate = float(line_in_config(config, 'timerate'))

log = []
data_log = []

r_avg = 0

res = saveResolution

def normalize_img(img):
    return np.divide(img, 256.0)
    

def denormailze_img(img):
    return np.rint(np.multiply(img, 256.0))

def img_2_list(img):
    return img.flatten().tolist()

def list_2_img(array):
    return np.array(array).reshape((res[1], res[0]))
    
def split_colors(img, n):
    return np.divide(np.round(np.multiply(img, n)), n)

def border_img(img):
    return np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0.5))

def gaussian_blur(img):
    array = []
    kernel = np.array([0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625])
    for x in range(1, res[1] + 1):
        for y in range(1, res[0] + 1):
            cut = img[x-1:x+2,:][:,y-1:y+2].flatten()
            array.append(np.dot(kernel, cut))
    return list_2_img(array)

def edge_detection(img):
    array_h = []
    array_v = []
    kernel_h = np.array([0.25, 0.5, 0.25, 0, 0, 0, -0.25, -0.5, -0.25])
    kernel_v = np.array([0.25, 0, -0.25, 0.5, 0, -0.5, 0.25, 0, -0.25])
    for x in range(1, res[1] + 1):
        for y in range(1, res[0] + 1):
            cut = img[x-1:x+2,:][:,y-1:y+2].flatten()
            array_h.append(np.dot(kernel_h, cut))
            array_v.append(np.dot(kernel_v, cut))
    array = []
    for i in range(len(array_v)):
        array.append(m.sqrt(array_h[i] ** 2 + array_v[i] ** 2))
    return list_2_img(array)

while True:
    t1 = time.clock()
    img = urllib.urlopen(url)
    image = np.asarray(bytearray(img.read()), dtype='uint8')
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, saveResolution)
    small = normalize_img(small)

    e = edge_detection(border_img(small))
    b = gaussian_blur(border_img(small))

    small = np.add(np.multiply(e, 0.75), np.multiply(b, 0.25))

    small = split_colors(small, 20.0)

    
    big = cv2.resize(small, dispResolution)
    log.append(small)
    
    cv2.imshow('f', big)

    
    MutexController.MutexWaitOne(mutex_name)
    dt = open(data_path, 'r')
    lns = []
    for l in dt:
        lns.append(l)
    dt.close()
    MutexController.MutexRelease(mutex_name)
    data_log.append([lns[0].replace('\n', '').replace(',', '.'), lns[1].replace('\n', '').replace(',', '.')])

    t2 = time.clock()
    dt = t2 - t1
    if dt < timerate:
        time.sleep(timerate - (dt))
    r_avg = (r_avg * 10 + (dt)) / 11
    print len(log), r_avg, dt, data_log[-1]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print len(log), len(data_log)

text_log = []
for i in range(len(log)):
    array = []
    for row in log[i]:
        for p in row:
            array.append(str(p / 256.0))
    
    text_log.append(','.join(array) + ',' + ','.join(data_log[i]))
"""
with open('no_log.txt', 'a') as file:
        file.write('\n'.join(text_log))
"""
