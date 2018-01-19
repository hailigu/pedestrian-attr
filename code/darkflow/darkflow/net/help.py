"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os
import csv
import subprocess as sp
import numpy
import re

from PIL import Image

old_graph_msg = 'Resolving old graph def {} (no guarantee)'


def ffmpeg_pipe(self, file, w, h):
    url_head = 'rtmp://video-center-bj.alivecdn.com/app/'
    # file might be absolute path, we just need filename
    file = os.path.basename(file)
    url_stream = re.split('\.', file)[0]   # get test from test.avi
    url_host = '?vhost=live.hailigu.com'
    url = '%s%s-%s%s' % (url_head, url_stream,self.FLAGS.object_id, url_host)
    size = '%dx%d' % (w, h)
    print (size)
    cmd_out1 = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', size,
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                url]
    self.pipe = sp.Popen(cmd_out1, stdin=sp.PIPE)

def build_train_op(self):
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    gradients = optimizer.compute_gradients(self.framework.loss)
    self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
    if self.FLAGS.load < 0: # load lastest ckpt
        with open(os.path.join(self.FLAGS.backup, 'checkpoint'), 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)

    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try: self.saver.restore(self.sess, load_point)
    except: load_old_graph(self, load_point)

def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)

def load_old_graph(self, ckpt):
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))

    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
        'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})

def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame)
    return timer() - start

def camera_stop(self):
    self.FLAGS.process_status = 1

def camera_pause(self):
    self.FLAGS.process_status = 2

def camera_resume(self):
    self.FLAGS.process_status = 0

def camera_get(self):
    return self.FLAGS.process_status

def camera_set(self, x1,y1,x2,y2):
    self.FLAGS.x1 = x1
    self.FLAGS.y1 = y1
    self.FLAGS.x2 = x2
    self.FLAGS.y2 = y2
def camera(self):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo

    if self.FLAGS.track :
        if self.FLAGS.tracker == "deep_sort":
            from deep_sort import generate_detections
            from deep_sort.deep_sort import nn_matching
            from deep_sort.deep_sort.tracker import Tracker
            metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 0.2, 100)
            tracker = Tracker(metric)
            encoder = generate_detections.create_box_encoder(
                os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
        elif self.FLAGS.tracker == "sort":
            from sort.sort import Sort
            encoder = None
            tracker = Sort()
    if self.FLAGS.BK_MOG and self.FLAGS.track :
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)

    camera = cv2.VideoCapture(file)

    if file == 0:
        self.say('Press [ESC] to quit video')

    assert camera.isOpened(), \
    'Cannot capture source'

    if self.FLAGS.csv :
        f = open('{}.csv'.format(file),'w')
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['frame_id', 'track_id' , 'x', 'y', 'w', 'h'])
        f.flush()
    else :
        f =None
        writer= None
    if file == 0:#camera window
        cv2.namedWindow(self.FLAGS.object_id, 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow(self.FLAGS.object_id, width*0.5, height*0.5)
    else:
        _, frame = camera.read()
        height, width, _ = frame.shape
    
    if self.FLAGS.push_stream:
        ffmpeg_pipe(self, file, width, height)

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame)
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            'output_{}'.format(file), fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()

    elapsed = 0
    start = timer()
    self.say('Press [ESC] to quit demo')
    #postprocessed = []
    # Loop through frames
    n = 0
    while camera.isOpened():
        if self.FLAGS.process_status == 1:  # esc
            print("gongjia: Stoped! " )
            break
        if self.FLAGS.process_status == 2:
            #print("gongjia: Paused! ")
            continue
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        if self.FLAGS.skip != n :
            n+=1
            continue
        n = 0
        if self.FLAGS.BK_MOG and self.FLAGS.track :
            fgmask = fgbg.apply(frame)
        else :
            fgmask = None
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                if not self.FLAGS.track :
                    postprocessed = self.framework.postprocess(
                        single_out, img)
                else :
                    postprocessed = self.framework.postprocess(
                        single_out, img,frame_id = elapsed,
                        csv_file=f,csv=writer,mask = fgmask,
                        encoder=encoder,tracker=tracker)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if self.FLAGS.display :
                    cv2.imshow(self.FLAGS.object_id, postprocessed)
                if self.FLAGS.push_stream:
                    #im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    self.pipe.stdin.write(postprocessed.tobytes())

            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(' {0:3.3f} FPS '.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if self.FLAGS.display :
            choice = cv2.waitKey(1)
            if choice == 27:
                break

    cv2.imwrite('{}_{}_counter.jpg'.format(self.FLAGS.demo, self.FLAGS.object_id), postprocessed)
    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    if self.FLAGS.csv :
        f.close()
    camera.release()
    if self.FLAGS.display :
        cv2.destroyAllWindows()
    if self.FLAGS.push_stream:
        self.pipe.stdin.close()
        self.pipe.wait()

def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt
