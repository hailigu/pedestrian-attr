# -*- coding: utf-8 -*-
# @Time    : 2018/1/4 22:07
# @Author  : DeepKeeper (DeepKeeper@qq.com)
# @Site    :
# @File    : app.py

import codecs
import os
from types import *
import click
from flask import Flask, current_app,request, redirect, url_for, render_template, abort, g, request_started, \
    request_finished
from flask_redis import FlaskRedis

from flask_cors import CORS

from common.common import *
from common.helper import *

# global vars
COUNTER = 0
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(APP_ROOT, 'logs')
VIDEO_ROOT = os.path.join(APP_ROOT, 'static', 'videos')
LOG_FILE_PATH = os.path.join(LOG_ROOT, datetime.datetime.now(TZ).strftime('%Y-%m-%d') + '.log')
STREAM_URL_PREFIX = 'rtmp://live.hailigu.com/app/'

# create an instance
app = Flask(__name__)
# just set this to true when debug
app.config['DEBUG'] = True  # False
# cross domain request will be no problem
CORS(app)
# preserved for celery and redis
# app.config['REDIS_HOST'] = 'localhost'
# app.config['REDIS_PORT'] = 6379
# app.config['REDIS_DB'] = 0
# redis_store = FlaskRedis(app)


# init or preload something before app start,
# e.g. db connection, redis or loading models
@app.before_first_request
def init():
    click.secho(u'preserved for something before first request', fg='green')
    # we should use correct log level for better performance
    # when deployed in production environment
    init_logger(LOG_FILE_PATH, logging.INFO)
    logger.info('app is starting now')


# this action will be executed before each request
# e.g. request counter
@app.before_request
def before_request_handler():
    # just a demo
    global COUNTER
    COUNTER += 1
    click.secho(u'current request counter {}'.format(COUNTER), fg='yellow')


# this action will be executed after each request
# e.g. modify response header or data
@app.after_request
def after_request_handler(response):
    # just a fake header avoiding security problem
    response.headers['Server'] = 'HaiLiGu Cluster'
    response.headers[
        'X-Powered-By'] = 'Servlet 2.4; JBoss-4.0.5.GA (build: CVSTag=Branch_4_0 date=201301162339)/Tomcat-5.5'
    return response


# app index
@app.route('/')
def index():
    videos_frames = list_videos_and_frames(VIDEO_ROOT, True)
    return render_template('index.html', videos=videos_frames)


# list videos
@app.route('/list_videos', methods=['GET', 'POST'])
def list_videos():
    videos_frames = list_videos_and_frames(VIDEO_ROOT, True)
    if not videos_frames:
        return make_result('', 'specified video directory does not exist.', 1)
    return make_result(videos_frames)


# extract video frame
@app.route('/extract_frame', methods=['GET', 'POST'])
def extract_frame():
    video_id = parse_args('vid')
    if not video_id:
        return make_result('', 'you must specify the video Id.', 2)

    # It will extract frame randomly when no frame index specified
    frame_index = parse_args('index', -1)

    video_path = get_video_path_by_id(video_id, VIDEO_ROOT)
    if not os.path.exists(video_path):
        return make_result('', 'request video does not exist.', 1)

    video_frame = extract_video_frame(video_path, frame_index, True)
    return make_result(video_frame)


# set line points before start to process video
@app.route('/set_points', methods=['GET', 'POST'])
def set_points():
    video_id = parse_args('vid')
    if not video_id:
        return make_result('', 'you must specify the video Id.', 2)

    points = parse_args('points', '')
    if not points:
        return make_result('', 'you must specify points to apply.', 2)

    video_path = get_video_path_by_id(video_id, VIDEO_ROOT)
    if not os.path.exists(video_path):
        return make_result('', 'request video does not exist.', 1)

    # parse points and apply
    real_points = points.split(',')
    if len(real_points) != 4:
        return make_result('', 'length of points posted is incorrect.', 1)

    points = set_line_points(video_path, real_points[0], real_points[1], real_points[2], real_points[3])
    return make_result(points)



# get line points from config file
@app.route('/get_points', methods=['GET', 'POST'])
def get_points():
    video_id = parse_args('vid')
    if not video_id:
        return make_result('', 'you must specify the video Id.', 2)

    video_path = get_video_path_by_id(video_id, VIDEO_ROOT)
    if not os.path.exists(video_path):
        return make_result('', 'request video does not exist.', 1)

    points = get_line_points(video_path)
    return make_result(points)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
