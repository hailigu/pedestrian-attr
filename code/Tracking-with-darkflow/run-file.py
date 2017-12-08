from darkflow.darkflow.defaults import argHandler #Import the default arguments
import os
from darkflow.darkflow.net.build import TFNet

import csv
from GetCsvColumn import CsvFile,EXCLUDE

FLAGS = argHandler()
FLAGS.setDefaults()

FLAGS.demo = "test.avi" # video file to use, or if camera just put "camera"
FLAGS.model = "darkflow/cfg/yolo.cfg" # tensorflow model
FLAGS.load = "darkflow/bin/yolo.weights" # tensorflow weights
FLAGS.threshold = 0.25 # threshold of decetion confidance (detection if confidance > threshold )
FLAGS.gpu = 1 #how much of the GPU to use (between 0 and 1) 0 means use cpu
FLAGS.track = True # wheither to activate tracking or not
FLAGS.trackObj = "person" # the object to be tracked
FLAGS.saveVideo = True #whether to save the video or not
FLAGS.BK_MOG = False # activate background substraction using cv2 MOG substraction,
                        #to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
                        # helps only when number of detection < 5, as it is still better than no detection.
FLAGS.tracker = "deep_sort" # wich algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
FLAGS.skip = 2 # how many frames to skipp between each detection to speed up the network
FLAGS.csv = True #whether to write csv file or not(only when tracking is set to True)
FLAGS.display = False # display the tracking or not

tfnet = TFNet(FLAGS)

tfnet.camera()
exit('Demo stopped, exit.')
