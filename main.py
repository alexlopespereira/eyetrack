import glob
import math

import cv2
import numpy as np
from screeninfo import get_monitors
from util import RingBuffer
import pyautogui

s0x, s0y = pyautogui.position()

def exp_function(x):
    alpha = 0.005
    beta = 0.4
    ret = beta * math.exp(alpha * x - 1)
    return ret


def calc_displacements(s0, s1, inc):
    if s0 == s1:
        return 0
    d = exp_function(abs(s0 - s1) * inc)
    if s1 < s0:
        d = -d
    return d


ringlen = 10
total_W = total_H = 0
turning_th = 1

for m in get_monitors():
    total_H += m.height
    total_W += m.width
print(total_H)
print(total_W)
cap = cv2.VideoCapture(0)
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


# while True:
#     _, frame = cap.read()
images = glob.glob('data/*.png')
for fname in images:
    frame = cv2.imread(fname)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (3, 3), None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK)
    destx = s0x
    desty = s0y
    if ret == True:
        m = cv2.mean(corners)
        xhistory.extend(np.array([m[1]]))
        yhistory.extend(np.array([m[0]]))
        if not firstx:
            dx = xhistory.get()
            first_pointx = np.nonzero(dx)[0][0]
            dispx = dx[-1] - dx[first_pointx]
            curr_dist_x = abs(dispx)
            if curr_dist_x + turning_th < last_dist_x:
                xhistory = RingBuffer(ringlen)
                firstx = True
                s0x = (s0x - curr_dist_x) if dispx > 0 else (s0x + curr_dist_x)
                last_dist_x = 0
            else:
                last_dist_x = curr_dist_x
            incdx = calc_displacements(s0x, curr_dist_x, incx)
            destx = s0x + incdx

        if not firsty:
            dy = yhistory.get()
            first_pointy = np.nonzero(dy)[0][0]
            dispy = dy[-1] - dy[first_pointy]
            curr_dist_y = abs(dispy)
            if curr_dist_y + turning_th < last_dist_y:
                yhistory = RingBuffer(ringlen)
                firsty = True
                s0y = (s0y - curr_dist_y) if dispy > 0 else (s0y + curr_dist_y)
                last_dist_y = 0
            else:
                last_dist_y = curr_dist_y
            incdy = calc_displacements(s0y, curr_dist_y, incy)
            desty = s0y + incdy


        adjx = max(0, min(destx, total_W-1))
        adjy = max(0, min(desty, total_H-1))
        print(destx, desty, adjx, adjy)
        # pyautogui.moveTo(adjx, adjy)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(frame, (3, 3), corners, ret)
        cv2.imshow("corners", img)

    cv2.imshow("Frame", frame)
    cv2.imshow("gray", gray)
    firstx = False
    firsty = False

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

