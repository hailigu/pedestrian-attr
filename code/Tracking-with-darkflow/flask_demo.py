#!/usr/bin/env python
__author__ = 'gongjia'

import gevent
import gevent.monkey
from gevent.pywsgi import WSGIServer

from Control_API import control_p,EXCLUDE

gevent.monkey.patch_all()

from flask import Flask, request, Response, render_template

app = Flask(__name__)

def event_stream():
    count = 0
    # init, add by gongjia
    handle_p = control_p("test.avi")
    while True:
        gevent.sleep(2)
        yield 'data: %s\n\n' % count
        print ('data: %s\n\n' % count)
        count += 1

        # for test, add by gongjia
        if count == 5:
            yield "start..........\n"
            print ("start..........\n")
            handle_p.start_p()


@app.route('/my_event_source')
def sse_request():
    return Response(
            event_stream(),
            mimetype='text/event-stream')

@app.route('/')
def page():
    return render_template('sse.html')

if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 8001), app)
    http_server.serve_forever()