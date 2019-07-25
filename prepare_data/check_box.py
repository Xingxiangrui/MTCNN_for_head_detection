#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:33:17 2018

@author: lzw

check the box
"""
import cv2
import numpy as np
fo = open("list.txt")
lines = fo.readlines()
for line in lines:
    line = line.split()
    img_path = line[0]
    bbox = [float(i) for i in line[1:]]
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(img_path)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
    cv2.imshow("test", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()