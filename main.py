import glob
import math

import cv2
import numpy as np
from screeninfo import get_monitors
from util import RingBuffer
import pyautogui
pyautogui.FAILSAFE = False

s0x, s0y = 400, 400 #pyautogui.position()


def calc_displacements(s, inc, axis=0):
    d = 0.9 * abs(s) * inc
    return d


ringlen = 1000
total_W = total_H = 0
turning_th = 2

for m in get_monitors():
    total_H += m.height
    total_W += m.width
print(total_H)
print(total_W)
cap = cv2.VideoCapture(2)
_, frame = cap.read()
h = frame.shape[0]
w = frame.shape[1]
incx = total_W/float(w)
incy = total_H/float(h)
startx = starty = 0
font = cv2.FONT_HERSHEY_PLAIN
xhistory = RingBuffer(ringlen)
yhistory = RingBuffer(ringlen)
last_dist_x = last_dist_y = 0
firstx = True
firsty = True
count = 0
data = np.array([[125, 249], [135, 259], [145, 269], [155, 279],
                 [135, 259], [115, 239], [ 95, 219], [ 75, 199]])
while True:
    _, frame = cap.read()
    # s0x, s0y = pyautogui.position()
# images = sorted(glob.glob('data/*.png'))
# for fname in images:
#     frame = cv2.imread(fname)
# for m in data:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (3, 3), None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK)
    destx = s0x
    desty = s0y
    # if True:
    if ret == True:
        m = cv2.mean(corners)
        xhistory.extend(np.array([m[0]]))
        yhistory.extend(np.array([m[1]]))
        count += 1
        if count < 12:
            continue

        dx = xhistory.get()
        dispx = np.mean(dx[-3:]) - np.mean(dx[-6:-3])
        last_dispx = np.mean(dx[-9:-6]) - np.mean(dx[-6:-3])
        changed_directionx = np.sign(dispx) != np.sign(last_dispx)
        if changed_directionx:
            tmp = dx[-2:]
            xhistory = RingBuffer(ringlen)
            xhistory.extend(np.array(tmp))
            dx = xhistory.get()
            dispx = dx[-1] - dx[-2]

        incdx = calc_displacements(dispx, incx, axis=0)
        signx = 1 if dispx > 0 else -1
        destx = int(s0x + incdx*signx)

        dy = yhistory.get()
        dispy = np.mean(dy[-3:]) - np.mean(dy[-6:-3])
        last_dispy = np.mean(dy[-9:-6]) - np.mean(dy[-6:-3])
        changed_directiony = np.sign(dispy) != np.sign(last_dispy)
        if changed_directiony:
            tmp = dy[-2:]
            yhistory = RingBuffer(ringlen)
            yhistory.extend(np.array(tmp))
            dy = yhistory.get()
            dispy = dy[-1] - dy[-2]

        incdy = calc_displacements(dispy, incy, axis=1)
        signy = 1 if dispy > 0 else -1
        desty = int(s0y+ incdy*signy)

        s0x = int(max(1, min(destx, total_W-10)))
        s0y = int(max(1, min(desty, total_H-10)))
        print(int(m[0]), int(m[1]), s0x, s0y)
        pyautogui.moveTo(total_W - s0x, s0y)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(frame, (3, 3), corners, ret)

    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

cap.release()
cv2.destroyAllWindows()

