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


