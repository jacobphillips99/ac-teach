import cv2
import numpy as np
import os

from os.path import isfile, join

frame_array = []
# inPath ='/home/jacobphillips/sandbox/ac-teach/videos/03-04-2020_11-09-09/12626/'
# inPath ='/home/jacobphillips/sandbox/ac-teach/videos/03-04-2020_11-09-09/12632/'
# inPath ='/home/jacobphillips/sandbox/ac-teach/videos/03-04-2020_11-09-09/56559/'
inPath ='/home/jacobphillips/sandbox/ac-teach/videos/03-04-2020_11-09-09/56562/'
files = [f for f in os.listdir(inPath) if f[-3:]=='png']
files.sort(key = lambda x: int(x[:-4]))

for filename in files:
    img = cv2.imread(os.path.join(inPath, filename))
    frame_array.append(img)
size = frame_array[0].shape[:3]
out = cv2.VideoWriter("video56k_2.avi", 0, 10, (500,500))


for frame in frame_array:
    out.write(frame)
out.release()
print("done")
