# -*- coding: utf-8 -*-
# @Time    : 2017/12/23 17:24
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    : 
# @File    : demo.py
# @Software: PyCharm

from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
from collections import namedtuple

import PIL
import cv2
import imutils
import mxnet as mx
import numpy as np
from sort import *


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def img_preprocessing(img, data_shape):
    img = cv2.resize(img, (data_shape, data_shape))

    mean = np.array([123, 117, 104])
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return [mx.nd.array([img])]


def get_color(c, x, max):
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (float(x) / max) * 5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1 - ratio) * colors[int(i)][c] + ratio * colors[int(j)][c]
    return r


def max_index(a, n):
    if n <= 0:
        return -1
    i = 0
    max_i = 0
    max = a[0]
    for i in range(n):
        if a[i] > max:
            max = a[i]
            max_i = i
    return max_i

colors_table = dict()

def get_bboxes(img, dets, thresh=0.5):
    height = img.shape[0]
    width = img.shape[1]
    bboxes = []
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = dets[i, 1]
            if score > thresh:
                if cls_id not in colors_table:
                    offset = int(score * 123457 % len(class_list))
                    red = get_color(2, offset, len(class_list)) * 255
                    green = get_color(1, offset, len(class_list)) * 255
                    blue = get_color(0, offset, len(class_list)) * 255
                    colors_table[cls_id] = (blue, green, red)
                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)
                bboxes.append([cls_id, score, xmin, ymin, xmax, ymax])
    return bboxes


parser = argparse.ArgumentParser(description="command")
parser.add_argument('--video', type=str, default='./33.mp4', help='the video input')
parser.add_argument('--frame-count', dest='frame_count', type=int, default=0,
                    help='frame count to be detected')
parser.add_argument('--gpu', dest='gpu_index', type=int, default=-1,
                    help='gpu id')

args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)
class_list = ['__nothing__', 'person', ]

model_ = 'deploy_ssd_resnet_people_3590'
prefix = './models/' + model_
epoch = 0
cls_id = 1

data_shape = 512
mean_pixels = (123, 117, 104)
batch_size = 1
context = mx.gpu(args.gpu_index)

#cv2.namedWindow("Video")
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
file_name, file_ext = get_file_name_and_ext(args.video)

sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
mod = mx.mod.Module(symbol=sym, context=context, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, data_shape, data_shape))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params=arg_params, aux_params=aux_params, force_init=False, allow_extra=True, allow_missing=True)

m_tracker = Sort(max_age=60, min_hits=1)

if size[0] > 1400 or size[1] > 1024:
    size = (int(size[0] * 0.5), int(size[1] * 0.5))

# rotation = True
rotation = False
if rotation:
    vw = cv2.VideoWriter(file_name + '_' + model_ + '_out.mp4', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 25,
                         (size[1], size[0]))
else:
    vw = cv2.VideoWriter(file_name + '_' + model_ + '_out.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 25,
                         (size[0], size[1]))

if cap is None:
    print("Error opening video file.")
    sys.exit(0)
if not vw.isOpened():
    print("Error opening video writer!")
    sys.exit(0)

k = 0
frame_count = 0
last_count = 0
avg_fps = 0
all_fps = 0

while True:

    if frame_count != 0 and frame_count == args.frame_count:
        print("configured number of frames written to video.")
        cap.release()
        vw.release()
        #cv2.destroyAllWindows()
        break

    (grabbed, frame) = cap.read()
    if not grabbed:
        print("End of video stream reached.")
        break

    if frame.shape[0] == 0:
        break

    if rotation:
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 0)

    o_height, o_width = frame.shape[:2]
    # print (o_height,o_width)

    if o_width > 1400 or o_height > 1024:
        size = (int(o_width * 0.5), int(o_height * 0.5))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    original_size = frame.shape[:2]
    # print(frame.shape)

    original_frame = np.zeros(frame.shape, np.uint8)
    original_frame = frame.copy()
    # print (origial_frame.shape)

    data = img_preprocessing(frame, data_shape)

    frame_count += 1

    det_batch = mx.io.DataBatch(data, [])
    start = time.time()
    mod.forward(det_batch, is_train=False)
    detections = mod.get_outputs()[0].asnumpy()
    time_elapsed = time.time() - start
    current_fps = 1 / time_elapsed
    all_fps += current_fps
    print("time: {:.4f}  current fps: {:.3f}  average fps:  {:.3f}".format(time_elapsed, current_fps,
                                                                           all_fps / frame_count))
    cv2.putText(original_frame, "time: {:.4f}  cfps: {:.3f}  afps:  {:.3f} ".format(time_elapsed, current_fps,
                                                                                    all_fps / frame_count),
                (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 1, 1)

    result = []
    for i in range(detections.shape[0]):
        det = detections[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]
        result.append(res)
    bboxes = get_bboxes(frame, res)

    for i in range(len(bboxes)):
        if bboxes[i][0] == cls_id:
            cv2.rectangle(original_frame, (bboxes[i][2], bboxes[i][3]), (bboxes[i][4], bboxes[i][5]),
                              colors_table[bboxes[i][0]], 2)

    dets = []
    for i in range(len(bboxes)):
        det_new = np.append(bboxes[i][2:6], bboxes[i][1])
        # print (det_new.shape)
        # print (det_new)
        dets.append(det_new)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    tracks = m_tracker.update(np.array(dets))
    for i in range(len(tracks)):
        # print (track_ids[i])
        cv2.putText(original_frame, '{0}'.format(int(tracks[i][4])),
                    (int(tracks[i][0]) + 4, max(0, int(tracks[i][3]) - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, 1)
        cv2.rectangle(original_frame, (int(tracks[i][0]), int(tracks[i][1])),
                      (int(tracks[i][2]), int(tracks[i][3])),
                      (255, 0, 0), 2)

    #cv2.imshow("Video", original_frame)
    vw.write(original_frame)

    k = cv2.waitKey(1)
    if k == 27:
        cap.release()
        vw.release()
        #cv2.destroyAllWindows()
        break

cap.release()
vw.release()
#cv2.destroyAllWindows()
