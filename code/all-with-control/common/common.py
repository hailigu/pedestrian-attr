# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 22:24
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    : 
# @File    : common.py
import os
import glob
import sys
import logging
import datetime
import pytz
import random
import uuid
from types import *
from flask import request, jsonify, json

from common.gpu import *
from common.cpu_ram import *

# timezone and time format
UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
TZ = pytz.timezone('Asia/Shanghai')

# just for logger
logging.basicConfig()
logger = logging.getLogger()


# In order to handle all situations friendly, we response data
# as json object. It means error occurred when code is NOT zero,
# then message will be its corresponding friendly message
# logId, serverId, finishedTime are preserved for tracking request
def make_result(result, message=None, code=0):
    # if result is not str:
    #     result = json.dumps(result)
    now_time = datetime.datetime.now(TZ).strftime(UTC_FORMAT)
    return jsonify({
        "meta": {
            "code": code,
            "message": message,
            "logId": str(uuid.uuid1()),
            "serverId": random.randint(1, 8),
            "finishedTime": now_time
        },
        "results": {
            "result": result
        }
    })


# helper to obtain value from request
def parse_args(arg_key, default_value=None, method='all'):
    if method.lower() == 'post':
        post_value = request.form.get(arg_key, default_value)
        return post_value
    elif method.lower() == 'get':
        get_value = request.args.get(arg_key, default_value)
        return get_value
    elif method.lower() == 'cookie':
        return request.cookies.get(arg_key, default_value)
    else:
        # generally we just wanna values by get and post
        get_value = request.args.get(arg_key, default_value)
        if not get_value:
            return request.form.get(arg_key, default_value)
        else:
            return get_value


# get file name and its extension name
def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


# get files within specified directory
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


# get sub directory within specified directory
def get_dirs(dir_name):
    dirs = []
    for root_dir, sub_dirs, files in os.walk(dir_name):
        for sub_dir in sub_dirs:
            dirs.append(os.path.join(root_dir, sub_dir))
    return dirs


# log or display some message
def show_message(message, log, stop=False):
    if stop:
        message += "\n\n\n\n"
    if log:
        log.info(message)
    else:
        print(message)
    if stop:
        sys.exit(0)


# parse string to bool
def str2bool(v):
    if not v:
        return False
    return v.lower() in ("yes", "true", "t", "1")


# init logger for apps
def init_logger(log_file, level=logging.INFO):
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    logger.addHandler(fh)


# make enum compatible with py2 and py3
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# query status and merge it into dictionary
def get_all_server_status():
    server_status = dict()

    gpu_status = get_gpu_status()
    ram_status = get_ram_status()
    load_status, _ = get_server_load()

    server_status['ram'] = ram_status
    server_status['load'] = load_status
    server_status['gpuload'] = '{:.1%}'.format(gpu_status.utilization / 100)
    server_status['gpuram'] = '{:.1%}'.format(gpu_status.memory_used/gpu_status.memory_total)
    return server_status
