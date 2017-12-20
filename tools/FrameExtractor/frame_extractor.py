# -*- coding: utf-8 -*-
# @Time    : 2017/12/14 22:23
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    :
# @File    : frame_extractor.py
import argparse
import glob
import logging
import math
import os
import random
import sys
import time
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


def show_message(message,log, stop=False):
    if stop:
        message += "\n\n\n\n"
    if log:
        log.info(message)
    else:
        print (message)
    if stop:
        sys.exit(0)


def frame_extractor(args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)
    if not os.path.isdir(args.source_dir):
        show_message('source_dir {} is incorrect'.format(args.source_dir),logger, True)
    if not os.path.exists(args.save_root_dir):
        os.mkdir(args.save_root_dir)
    files = get_files(args.source_dir,args.search_pattern,args.recursive)
    if len(files) == 0:
        show_message("there's no {} file in directory {} ".format(args.search_pattern,args.source_dir),logger, True)
    for f in files:
        show_message('video  path {}'.format(os.path.dirname(f)), logger)
        reader = cv2.VideoCapture(f)

        if not reader :
            show_message('error opening video {}'.format(os.path.basename(f)),logger)
            continue
        show_message('video to be extracted  {}'.format(os.path.basename(f)),logger)

        file_name, file_ext = get_file_name_and_ext(f)
        start_index = 0
        end_index = 0
        original_frame_count = 0

        if file_ext.lower() == '.h264':
            show_message('fetching {} frame count, It may take some time'.format(os.path.basename(f)), logger)
            while (True):
                (grabbed, frame) = reader.read()
                if not grabbed:
                    show_message('End of video stream reached', logger)
                    break
                original_frame_count +=1
            show_message('video {} has {} frames'.format(os.path.basename(f),original_frame_count), logger)
        else:
            # fps = reader.get(cv2.CAP_PROP_FPS)
            # size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            original_frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if original_frame_count <= 0:
                show_message('file {} is not a valid video, skipped\n\n\n\n'.format(os.path.basename(f)), logger)
                continue
            show_message('original frame count  {}'.format(original_frame_count), logger)

        extracted_count = int(math.ceil(original_frame_count * args.extract_ratio))

        if args.skip_ratio > 0:
            frame_count_after_skip = int(math.ceil(original_frame_count * (1 - args.skip_ratio * 2)))
            start_index = int(math.floor(original_frame_count * args.skip_ratio)) - 1
            end_index = original_frame_count - int(math.floor(original_frame_count * args.skip_ratio)) - 1
            show_message('frame count after skipped  {}'.format(frame_count_after_skip),logger)
            if frame_count_after_skip < extracted_count:
                show_message('value of skip-radio is too big {} for '.format(os.path.basename(f)), logger)
        else:
            end_index = original_frame_count - 1

        show_message('frame start to be extracted  {}'.format(start_index),logger)
        show_message('frame end to be extracted  {}'.format(end_index),logger)
        show_message('frame count to be extracted  {}'.format(extracted_count),logger)
        frame_indexes = sorted(random.sample(np.arange(start_index, end_index), extracted_count))
        # print (frame_indexes)
        save_path = os.path.join(args.save_root_dir, os.path.basename(file_name))

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if file_ext.lower() == '.h264':
            index_count = 0
            reader = cv2.VideoCapture(f)
            while (True):
                (grabbed, frame) = reader.read()
                if not grabbed:
                    show_message('End of video stream reached', logger)
                    break
                if frame_indexes.count(index_count) > 0:
                    cv2.imwrite(os.path.join(save_path, '{}_{:06d}.jpg'.format(file_name, index_count)), frame)
                    show_message('extracting from {} at index {}'.format(os.path.basename(f), index_count), logger)
                index_count += 1
        else:
            for frame_index in frame_indexes:
                # https://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#cv.SetCaptureProperty
                reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                # reader.set(cv2.CAP_PROP_POS_MSEC, frame_index)
                (grabbed, frame) = reader.read()
                if not grabbed:
                    show_message('End of video stream reached', logger)
                cv2.imwrite(os.path.join(save_path, '{}_{:06d}.jpg'.format(file_name, frame_index)), frame)
                show_message('extracting from {} at index {}'.format(os.path.basename(f), frame_index), logger)

        reader.release()
        show_message('finished extracting from {} \n\n\n\n'.format(os.path.basename(f)), logger)


def parse_args():
    parser = argparse.ArgumentParser(description="video frame extractor by DeepKeeper ")
    parser.add_argument('--source-dir', dest='source_dir', type=str, default='./video/2',
                        help='the video source directory')
    parser.add_argument('--save-dir', dest='save_root_dir', type=str, default='./save',
                        help='the directory extracted frame to be saved')
    parser.add_argument('--ratio', dest='extract_ratio', type=float, default=0.1,
                        help='indicate frame count to be extracted divided by video frame count')
    parser.add_argument('--skip-ratio', dest='skip_ratio', type=float, default=0.0,
                        help='indicate frame count to be skipped at the start and the end of video divided by video '
                             'frame count')
    parser.add_argument('--recursive', dest='recursive', type=bool, default=True,
                        help='search videos in sub directory')
    parser.add_argument('--search-pattern', dest='search_pattern', type=str, default='*.*',
                        help='search pattern for video files *.*')
    parser.add_argument('--log', dest='log_file', type=str, default="log.txt",
                        help='save log to file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    frame_extractor(args)
