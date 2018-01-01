# coding=utf-8
import os
import sys

import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import time
import argparse

classes_coco = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
                'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

allow_index_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                    '15', '16',
                    '24', '25', '26', '28',
                    '57', '58', '59', '60', '61',
                    '63', '65', '66', '67', '68',
                    '73']

parser = argparse.ArgumentParser(description="command")
parser.add_argument('--video', type=str, default='./33.mp4', help='the video input')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
cv2.namedWindow("Video", flags=cv2.WINDOW_NORMAL)

fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_count = 0
avg_fps = 0
all_fps = 0

basename = os.path.basename(args.video)
model_dir = './models/'

# model_name = 'ssd_inception_v2'
model_name = 'ssd_mobilenet_v1'

model_path = model_dir + model_name
vw = cv2.VideoWriter('{}_{}.avi'.format(basename, model_name), cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, size, 1)

pb_path = model_path + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            if cap.isOpened():
                ret, image_np = cap.read()
                if not ret:
                    print("End of video stream reached.")
                    break
                height, width = image_np.shape[:2]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                start = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                time_elapsed = time.time() - start
                current_fps = 1 / time_elapsed
                all_fps += current_fps
                frame_count += 1
                print("time: {:.6f}  current fps: {:.3f}  average fps:  {:.3f}".format(time_elapsed, current_fps,
                                                                                       all_fps / frame_count))

                for box_group in boxes:
                    counter = 0
                    # box => ymin xmin ymax xmax
                    for box in box_group:
                        if counter > 50:
                            break
                        if classes[0][counter] != 1:
                            counter += 1
                            continue
                        if int(box[3] * width) - int(box[1] * width) > width / 1:
                            counter += 1
                            continue
                        if float(scores[0][counter]) < 0.48:
                            counter += 1
                            continue
                        if classes[0][counter] >= 80:
                            continue
                        class_index = int(classes[0][counter]) - 1
                        if allow_index_list.count(str(class_index)) <= 0:
                            continue

                        cv2.rectangle(image_np, (int(box[1] * width), int(box[0] * height)),
                                      (int(box[3] * width), int(box[2] * height)), (255, 0, 255), 2)
                        class_index = int(classes[0][counter]) - 1
                        cv2.putText(image_np, '{0} {1:.3f}'.format(classes_coco[class_index], scores[0][counter]),
                                    (int(box[1] * width) + 2, int(box[0] * height) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 1)
                        counter += 1

                cv2.imshow('Video', image_np)
                vw.write(image_np)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cap.release()
                    vw.release()
                    cv2.destroyAllWindows()
                    break
