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





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
