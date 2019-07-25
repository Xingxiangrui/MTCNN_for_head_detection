# -*- coding: utf-8 -*-
"""
遍历文件
This is a temporary script file.
x y w h
"""
from PIL import Image
# im = Image.open(filename)
import os
import shutil
oriDri = "NFPA_data"
newDir = "NFPA_mtcnn"
#width = 352 
#height = 288
fw = open('list.txt','w')
for root,dirs,files in os.walk(oriDri):
    for file in files:
        if 'jpg' in file:
            # copy jpg
            oldPic = root + os.path.sep + file
            newPic = newDir + os.path.sep + file
            shutil.copyfile(oldPic,newPic)
            
            fw.write(newPic)
            im = Image.open(oldPic)
            width = im.size[0]
            height = im.size[1]
            # write txt
            oldTxt = root + os.path.sep + file.replace('jpg','txt')
            fo = open(oldTxt,'r')
            lines = fo.readlines()
            for line in lines:
                line = line.split()
                x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                #[x, y, w, h] = [float(line[i]) for i in range(1,5)]
                x1 = round(x * width, 2)
                y1 = round(y * height, 2)
                x2 = round(x1 + w * width, 2)
                y2 = round(y1 + h * height, 2)
                print(x1,y1,x2,y2)
                fw.write(" " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
            fw.write("\n")
            fo.close()
            #print("fine")
print("done")
fw.close()
