import numpy as np
import cv2
import urllib
import time
import os.path
import clr
from PyMutexController import MutexController
import oct2py

def line_in_config(cnf, name):
    for l in cnf:
        spt = l.split('>')
        if spt[0] == name:
            return spt[1].replace('\n', '')

def str_2_tuple(s):
    sp = s.split(',')
    return (int(sp[0]), int(sp[1]))

config = []

f = open('driver.config', 'r')
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

octave = oct2py.Oct2Py()

print '\\'.join(__file__.split('\\')[1:-1])
print octave.cd('C:\\' + '\\'.join(__file__.split('\\')[1:-1]))

print 'load theta files'
#ThetaSteer = octave.load('Theta_steer.txt')
#ThetaSpeed = octave.load('Theta_speed.txt')

Theta = octave.load('Thetas.txt')


while True:
    t1 = time.clock()
    img = urllib.urlopen(url)
    image = np.asarray(bytearray(img.read()), dtype='uint8')
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, saveResolution)
    big = cv2.resize(small, dispResolution)
        

    cv2.imshow('f', big)

    array = []
    for row in small:
        for p in row:
            array.append(p / 256.0)

    x = np.array(array, dtype='float')

    #steer, speed = octave.evaluateLinearDriving(ThetaSpeed['Theta_speed'], ThetaSteer['Theta_steer'], x)
    #print Theta
    steer = (octave.threeLayerPredict(Theta['Thetas'], 600, 133, 31, x) - 1) / 15.0 - 1
    
    MutexController.MutexWaitOne(mutex_name)
    dt = open(data_path, 'w')
    dt.write(str(steer))
    dt.close()
    MutexController.MutexRelease(mutex_name)
    
    t2 = time.clock()
    dt = t2 - t1
    if dt < timerate:
        time.sleep(timerate - (dt))
    r_avg = (r_avg * 10 + (dt)) / 11
    print r_avg, dt, steer, speed

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
