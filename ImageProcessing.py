import numpy as np
import cv2
import urllib
import time
import os.path
import clr
import math as m

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
timerate = float(line_in_config(config, 'timerate'))

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


"""
img = urllib.urlopen(url)
image = np.asarray(bytearray(img.read()), dtype='uint8')
frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
small = cv2.resize(gray, saveResolution)

norm = normalize_img(small)
"""

log = []
f = open('log.txt', 'r')
for l in f:
    spl = l.replace('\n', '').split(',');
    ls = spl[0:-2];
    for i in range(len(ls)):
        ls[i] = float(ls[i])
    i = list_2_img(ls)
    e = edge_detection(border_img(i))
    b = gaussian_blur(border_img(i))
    ratio = 0.3
    i = np.add(np.multiply(e, 1 - ratio), np.multiply(b, ratio))
    i = split_colors(i, 20.0)
    li = img_2_list(i)
    for i in range(len(li)):
        ls[i] = str(li[i])
    ls.append(spl[-2])
    ls.append(spl[-1])
    log.append(ls)
f.close()

print log[552]

logS = []
for l in log:
    logS.append(','.join(l))

df = open('processed_log.txt', 'w')
df.write('\n'.join(logS))
df.close()

"""
ratio = 0.3

i = list_2_img(log[0])


e = edge_detection(border_img(i))
b = gaussian_blur(border_img(i))

i = np.add(np.multiply(e, 1 - ratio), np.multiply(b, ratio))

i = split_colors(i, 20.0)


big = cv2.resize(i, dispResolution)

            
cv2.imshow('f', big)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
"""
