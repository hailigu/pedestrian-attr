# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 0:24
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    : 
# @File    : helper.py

import os
import configparser as config
from types import *
import cv2
import numpy as np

from .common import *
from flask import url_for
from Control_API import *

# place to cache instance object
CACHED_OBJECT = {}


# extract specified frame in video when frame_index >=0
# -1 frame will be extracted randomly when video is seekable
def extract_video_frame(file_name, frame_index=-1, for_url=False):
    f_name, _ = get_file_name_and_ext(file_name)
    reader = cv2.VideoCapture(file_name)
    if not reader:
        logger.error('error opening video {}'.format(os.path.basename(file_name)))

    video_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_frame_count <= 0:
        logger.error('file {} is not a seekable video, skipped\n\n\n\n'.format(os.path.basename(file_name)))

    if frame_index == - 1:
        frame_index = np.random.randint(0, original_frame_count - 1)
    else:
        frame_index = min(frame_index, original_frame_count - 1)

    reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    (grabbed, frame) = reader.read()
    if not grabbed:
        logger.error('can not extract frame from {} at index {} .'.format(os.path.basename(file_name), frame_index))
        return 'not-seekable.jpg'

    frame_image_path = os.path.join(os.path.dirname(file_name), '{}.jpg'.format(f_name))
    # let's try golden ratio in case for too big frame image
    if video_width >1280 or video_height > 720:
        frame = cv2.resize(frame, (int(video_width*0.6180339887), int(video_height*0.6180339887)))
    cv2.imwrite(frame_image_path, frame)
    logger.info('finished extracting from {} at index {} .'.format(os.path.basename(file_name), frame_index))

    if for_url:
        frame_image_path = url_for('static', filename='videos/' + os.path.basename(frame_image_path))
    reader.release()
    return frame_image_path


# get videos and  frames within specified directory
def list_videos_and_frames(dir_name, for_url=False):
    if not os.path.isdir(dir_name):
        return False
    videos_frames = {}
    files = get_files(dir_name, '*.mp4')
    for file in files:
        file_name, file_ext = get_file_name_and_ext(file)
        frame_image_path = os.path.join(dir_name, file_name + '.jpg')
        if os.path.exists(frame_image_path):
            if for_url:
                videos_frames[file_name] = url_for('static', filename='videos/' + file_name + '.jpg')
            else:
                videos_frames[file_name] = file_name + '.jpg'
        else:
            frame_image_path = extract_video_frame(file, for_url=for_url)
            videos_frames[file_name] = frame_image_path
    return videos_frames


# set line points obtained from front page
def set_line_points(file_name, x1, y1, x2, y2):
    cap = cv2.VideoCapture(file_name)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    f_name, f_ext = get_file_name_and_ext(file_name)

    # transfer relative to absolute
    x1 = int(float(x1)*video_width)
    y1 = int(float(y1)*video_height)
    x2 = int(float(x2)*video_width)
    y2 = int(float(y2)*video_height)

    # print(x1, y1, x2, y2)
    conf = config.ConfigParser()
    points_conf_path = os.path.join(os.path.dirname(file_name), '{}_point.ini'.format(f_name))

    # we also write width and height
    if os.path.exists(points_conf_path):
        conf.read(points_conf_path)
        if conf.has_option('video', 'points'):
            conf.set('video', 'points', '{},{},{},{}'.format(x1, y1, x2, y2))
            conf.set('video', 'size', '{},{}'.format(video_width, video_height))
        else:
            conf.add_section('video')
            conf.set('video', 'points', '{},{},{},{}'.format(x1, y1, x2, y2))
            conf.set('video', 'size', '{},{}'.format(video_width, video_height))
    else:
        conf.add_section('video')
        conf.set('video', 'points', '{},{},{},{}'.format(x1, y1, x2, y2))
        conf.set('video', 'size', '{},{}'.format(video_width, video_height))

    with open(points_conf_path, 'w') as configfile:
        conf.write(configfile)
    return x1, y1, x2, y2



# just get points, should be invoked after set_line_points
def get_line_points(file_name):
    f_name, f_ext = get_file_name_and_ext(file_name)
    conf = config.ConfigParser()
    points_conf_path = os.path.join(os.path.dirname(file_name), '{}_point.ini'.format(f_name))
    if not os.path.exists(points_conf_path):
        return False
    conf.read(points_conf_path)
    points = conf.get('video', 'points')
    real_points = points.split(',')
    return int(real_points[0]), int(real_points[1]), int(real_points[2]), int(real_points[3])


# currently just hard code
def get_video_path_by_id(vid, video_root):
    return os.path.join(video_root, vid + '.mp4')


# currently just hard code
def get_video_frame_by_id(vid, video_root):
    return '/static/videos/' + vid + '.jpg'


