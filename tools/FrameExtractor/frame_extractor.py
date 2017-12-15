# -*- coding: utf-8 -*-
# @Time    : 2017/12/14 22:23
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    :
# @File    : frame_extractor.py
import argparse
import glob
import os
import sys
import time
import math
import random
from collections import namedtuple

import cv2
import numpy as np


def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def get_files(dir, file_type='*.*', recursive=True):
    all_files = []
    if dir:
        dir = dir.strip()
    if not os.path.isabs(dir):
        dir = os.path.abspath(dir)
    des_dir = os.path.join(dir, file_type)
    for file in glob.glob(des_dir):
        all_files.append(file)
    if recursive:
        sub_dirs = get_dirs(dir)
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(sub_dir, file_type)
            for file in glob.glob(sub_dir):
                all_files.append(file)
    return sorted(all_files)


def get_dirs(dir):
    dirs = []
    for root_dir, sub_dirs, files in os.walk(dir):
        for sub_dir in sub_dirs:
            dirs.append(os.path.join(root_dir, sub_dir))
    return dirs


def show_message(message, stop=False):
    print (message)
    if stop:
        sys.exit(0)


def frame_extractor(args):
    if not os.path.isdir(args.source_dir):
        show_message('source_dir {} is incorrect'.format(args.source_dir), True)
    if not os.path.exists(args.save_root_dir):
        os.mkdir(args.save_root_dir)
    files = get_files(args.source_dir,args.search_pattern,args.recursive)
    if len(files) == 0:
        show_message("there's no {} file in directory {} ".format(args.search_pattern,args.source_dir), True)
    for f in files:
        show_message('video to be extracted  {}'.format(os.path.basename(f)))
        reader = cv2.VideoCapture(f)
        if reader is None:
            show_message('error opening video {}'.format(os.path.basename(f)), True)
        # fps = reader.get(cv2.CAP_PROP_FPS)
        # size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        original_frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        show_message('original frame count  {}'.format(original_frame_count))
        start_index = 0
        end_index = original_frame_count - 1
        if args.skip_ratio:
            frame_count_after_skip = int(math.ceil(original_frame_count * (1 - args.skip_ratio * 2)))
            start_index = int(math.floor(original_frame_count * args.skip_ratio)) - 1
            end_index = original_frame_count - int(math.floor(original_frame_count * args.skip_ratio)) - 1
            show_message('frame count after skipped  {}'.format(frame_count_after_skip))
        extracted_count = int(math.ceil(original_frame_count * args.extract_ratio))
        if end_index - start_index < extracted_count:
            show_message('value of skip-radio is too big', True)
        show_message('frame start to be extracted  {}'.format(start_index))
        show_message('frame end to be extracted  {}'.format(end_index))
        show_message('frame count to be extracted  {}'.format(extracted_count))
        frame_indexes = sorted(random.sample(np.arange(start_index, end_index), extracted_count))
        # print (frame_indexes)
        show_message('start to extract frame from{}'.format(os.path.basename(f)))
        file_name, _ = get_file_name_and_ext(f)
        save_path = os.path.join(args.save_root_dir, os.path.basename(file_name))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for frame_index in frame_indexes:
            # https://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#cv.SetCaptureProperty
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            # reader.set(cv2.CAP_PROP_POS_MSEC, frame_index)
            (grabbed, frame) = reader.read()
            if not grabbed:
                show_message('End of video stream reached')
            cv2.imwrite(os.path.join(save_path, '{}_{}.jpg'.format(file_name, frame_index)), frame)
            show_message('extracting from {} at index {}'.format(file_name, frame_index))
        reader.release()


def parse_args():
    parser = argparse.ArgumentParser(description="video frame extractor by DeepKeeper ")
    parser.add_argument('--source-dir', dest='source_dir', type=str, default='./video/2',
                        help='the video source directory')
    parser.add_argument('--save-dir', dest='save_root_dir', type=str, default='./save',
                        help='the directory extracted frame to be saved')
    parser.add_argument('--ratio', dest='extract_ratio', type=float, default=0.1,
                        help='indicate frame count to be extracted divided by video frame count')
    parser.add_argument('--skip-ratio', dest='skip_ratio', type=float, default=0.0,
                        help='indicate frame count to be skipped at the start and the end of video divided by video frame count')
    parser.add_argument('--recursive', dest='recursive', type=bool, default=True,
                        help='search videos in sub directory')
    parser.add_argument('--search-pattern', dest='search_pattern', type=str, default='*.mp4',
                        help='search pattern for video files *.mp4 *.*')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    frame_extractor(args)
