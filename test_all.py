"""
author:  xing xiangrui
time  :  2018.9.20  20:30

predict all images in mAP/images
and store predict result in the VOC format

<class> <confidence> <left> <top> <right> <bottom>
in .txt in mAP/predicted

"""

import sys
import argparse
import time

import os  
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import cv2
import numpy as np
import glob

from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames


def main(args):


    file_paths = get_model_filenames(args.model_dir)
    
    #get image lise
    jpg_list = glob.glob(r'mAP/images/*.jpg')
    if len(jpg_list) == 0:
        print("Error: no .jpg files found in ground-truth")
    
    
    with tf.device('/gpu:2'):
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                print("LOCATION!!!tf config done"+"\n")
                if len(file_paths) == 3:
                    print("LOCATION!!!file_paths(model_dir)=3"+"\n")
                    image_pnet = tf.placeholder(
                        tf.float32, [None, None, None, 3])
                    pnet = PNet({'data': image_pnet}, mode='test')
                    out_tensor_pnet = pnet.get_all_output()

                    image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                    rnet = RNet({'data': image_rnet}, mode='test')
                    out_tensor_rnet = rnet.get_all_output()

                    image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                    onet = ONet({'data': image_onet}, mode='test')
                    out_tensor_onet = onet.get_all_output()
                    
                    print("LOCATION!!!placeholder and out_tensor done"+"\n")

                    saver_pnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "pnet/"])
                    saver_rnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "rnet/"])
                    saver_onet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "onet/"])

                    saver_pnet.restore(sess, file_paths[0])
                    
                    print("LOCATION!!!saver done"+"\n")

                    def pnet_fun(img): return sess.run(
                        out_tensor_pnet, feed_dict={image_pnet: img})

                    saver_rnet.restore(sess, file_paths[1])

                    def rnet_fun(img): return sess.run(
                        out_tensor_rnet, feed_dict={image_rnet: img})

                    saver_onet.restore(sess, file_paths[2])

                    def onet_fun(img): return sess.run(
                        out_tensor_onet, feed_dict={image_onet: img})
                    print("LOCATION!!!def net_fun done"+"\n")

                else:
                    print("LOCATION!!!ifile_paths(model_dir)!=3"+"\n")
                    saver = tf.train.import_meta_graph(file_paths[0])
                    saver.restore(sess, file_paths[1])

                    def pnet_fun(img): return sess.run(
                        ('softmax/Reshape_1:0',
                         'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                    def rnet_fun(img): return sess.run(
                        ('softmax_1/softmax:0',
                         'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                    def onet_fun(img): return sess.run(
                        ('softmax_2/softmax:0',
                         'onet/conv6-2/onet/conv6-2:0',
                         'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})
    
                
#                third_idxtry=[110,120]
#                for third_idx in third_idxtry:
                ROI_idx=[0,300,40,310]
                for tmp_file in jpg_list:
                    img=cv2.imread(tmp_file)
                    # add ROI region
                    ROI=img[ROI_idx[0]:ROI_idx[1],ROI_idx[2]:ROI_idx[3]]
                    ROI_temp=ROI.copy()
                    img[:,:,:]=0
                    img[ROI_idx[0]:ROI_idx[1],ROI_idx[2]:ROI_idx[3]]=ROI_temp
                    #create txt file
                    tmp_file=tmp_file.replace("jpg","txt")
                    txt_filename=tmp_file.replace("images","predicted")
                    print("LOACTION!!!predict:"+tmp_file)
                    
#                    start_time = time.time()
                    #print("LOCATION!!!detect_face function start"+"\n")
                    rectangles, points = detect_face(img, args.minsize,
                                                     pnet_fun, rnet_fun, onet_fun,
                                                     args.threshold, args.factor)
                    #print("LOCATION!!!idetect_face function done"+"\n")
#                    duration = time.time() - start_time
    
#                    print("duration:"+str(duration))
                    #print(type(rectangles))
                    points = np.transpose(points)
                    #print("LOCATION!!!loop rectangles"+"\n")
                    with open(txt_filename,'w') as result_file:
                        for rectangle in rectangles:
                            result_file.write("head" + " " + str(rectangle[4]) + " " + str(rectangle[0]) + " " + str(rectangle[1]) + " " + str(rectangle[2]) +  " " + str(rectangle[3])+"\n")
                    #print("LOCATION!!!Write done!"+"\n")
                print(ROI_idx)
                os.chdir("mAP/")
                os.system("python main.py -na")
    

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str,
                        help='The image path of the testing image')
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./save_model/all_in_one/')
    parser.add_argument(
        '--threshold',
        type=float,
        nargs=3,
        help='Three thresholds for pnet, rnet, onet, respectively.',
        default=[0.8, 0.8, 0.8])
    parser.add_argument('--minsize', type=int,
                        help='The minimum size of face to detect.', default=30)
    parser.add_argument('--factor', type=float,
                        help='The scale stride of orginal image', default=0.9)
    parser.add_argument('--save_image', type=bool,
                        help='Whether to save the result image', default=True)
    parser.add_argument('--save_name', type=str,
                        help='If save_image is true, specify the output path.',
                        default='result.jpg')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
