# -*- coding: utf-8 -*-
"""
author:xing xiangrui
time  :2018.9.26   10:34
copy image and change name to the new directory
write image name and labels to the list.txt

"""

import os
import shutil
oldDir = "data"
newDir = "head_data"
width = 352 
height = 288
fw = open('list.txt','w')
for root,dirs,files in os.walk(oldDir):
    if 'images' in root:
        continue
    for file in files:
        if 'jpg' in file:
            # copy jpg
            oldPic = root + os.path.sep + file
            newPic = newDir + os.path.sep + root.split(os.path.sep)[-2] + "_" + root.split(os.path.sep)[-1] + "_" + file
            shutil.copyfile(oldPic,newPic)
            
            fw.write(newPic)
            # write txt
            oldTxt = root + os.path.sep + file.replace('jpg','txt')
            fo = open(oldTxt,'r')
            lines = fo.readlines()
            for line in lines:
                line = line.split()
                cx, cy, w, h = float(line[1])*width, float(line[2])*height, float(line[3])*width, float(line[4])*height
                #[x, y, w, h] = [float(line[i]) for i in range(1,5)]
                x1 = round(cx - w/2, 2)
                y1 = round(cy - h/2, 2)
                x2 = round(cx + w/2, 2)
                y2 = round(cy + h/2, 2)
                print(x1,y1,x2,y2)
                fw.write(" " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
            fw.write("\n")
            fo.close()
            #print("fine")
print("done")
fw.close()