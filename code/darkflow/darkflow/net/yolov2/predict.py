import numpy as np
import math
import cv2
import os
import json
import csv
from GetCsvColumn import CsvFile,EXCLUDE

#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor


ds = True
csvfilename = 'test.avi.csv'
csvfile = CsvFile(csvfilename)

llx1 = 547
llx2 = 1251
lly1 = 825
lly2 = 964
ids = []

try :
	from deep_sort.application_util import preprocessing as prep
	from deep_sort.application_util import visualization
	from deep_sort.deep_sort.detection import Detection
except :
	ds = False


def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes


def extract_boxes(new_im):
    cont = []
    new_im=new_im.astype(np.uint8)
    ret, thresh=cv2.threshold(new_im, 127, 255, 0)
    p, contours, hierarchy=cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt=contours[i]
        x, y, w, h=cv2.boundingRect(cnt)
        if w*h > 30**2 and ((w < new_im.shape[0] and h <= new_im.shape[1]) or (w <= new_im.shape[0] and h < new_im.shape[1])):
            cont.append([x, y, w, h])
    return cont
def update_csv(count):
	with open(csvfilename, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		column = [row['track_id'] for row in reader]
		blist = list(set(column))
		clist = sorted([int(i) for i in blist])
	if count == 0:
		return len(clist)
	else:
		return count #clist.index(count) + 1

def postprocess(self,net_out, im,frame_id = 0,csv_file=None,csv=None,mask = None,encoder=None,tracker=None):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	nms_max_overlap = 0.1
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	thick = int((h + w) // 300)
	resultsForJSON = []

	if not self.FLAGS.track :
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if self.FLAGS.json:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
				continue
			if self.FLAGS.display or self.FLAGS.saveVideo:
				cv2.rectangle(imgcv,
					(left, top), (right, bot),
					colors[max_indx], thick)
				cv2.putText(imgcv, mess, (left, top - 12),
					0, 1e-3 * h, colors[max_indx],thick//3)
	else :
		if not ds :
			print("ERROR : deep sort or sort submodules not found for tracking please run :")
			print("\tgit submodule update --init --recursive")
			print("ENDING")
			exit(1)
		detections = []
		scores = []
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if self.FLAGS.trackObj != mess :
				continue
			if self.FLAGS.tracker == "deep_sort":
				detections.append(np.array([left,top,right-left,bot-top]).astype(np.float64))
				scores.append(confidence)
			elif self.FLAGS.tracker == "sort":
				detections.append(np.array([left,top,right,bot]).astype(np.float64))
		if len(detections) < 5  and self.FLAGS.BK_MOG:
			detections = detections + extract_boxes(mask)
		detections = np.array(detections)
		if self.FLAGS.tracker == "deep_sort":
			scores = np.array(scores)
			features = encoder(imgcv, detections.copy())
			detections = [
			            Detection(bbox, score, feature) for bbox,score, feature in
			            zip(detections,scores, features)]
			# Run non-maxima suppression.
			boxes = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			indices = prep.non_max_suppression(boxes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]
			tracker.predict()
			tracker.update(detections)
			trackers = tracker.tracks
		elif self.FLAGS.tracker == "sort":
			trackers = tracker.update(detections)

		for track in trackers:
			if self.FLAGS.tracker == "deep_sort":
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				bbox = track.to_tlbr()
				id_num = str(track.track_id)
			elif self.FLAGS.tracker == "sort":
				bbox = [int(track[0]),int(track[1]),int(track[2]),int(track[3])]
				id_num = str(int(track[4]))
			if self.FLAGS.csv:
				csv.writerow([frame_id,id_num,int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])])
				csv_file.flush()
			if self.FLAGS.display or self.FLAGS.saveVideo:
				#cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
				#		        (255,255,255), thick//3)
				#cv2.putText(imgcv, id_num,(int(bbox[0]), int(bbox[1]) - 12),0, 1e-3 * h, (255,255,255),thick//6)

				if id_num not in ids:

					newone = False
					ii = 0  

					while ii < 10: # Add a colon  
						nx = llx1 + (llx2 - llx1) * ii/ 10. 
						ny = lly1 + (lly2 - lly1) * ii/ 10. 
						minx = min(bbox[0], bbox[2])
						maxx = max(bbox[0], bbox[2])
						miny = min(bbox[1], bbox[3])
						maxy = max(bbox[1], bbox[3])
						if nx >= minx  and nx <= maxx and ny >= miny and ny <= maxy:
							newone = True;
							break
						ii += 1  

					if newone:
						ids.append(id_num)
				cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
						        (0,255,0), thick//3)
				cv2.putText(imgcv, str(update_csv(int(id_num))), (int(bbox[0]), int(bbox[1]) - 12), 0, 1e-3 * h, 							(255, 0, 255), thick // 6)
				font = cv2.FONT_HERSHEY_TRIPLEX
				#mycount = update_csv(0)
				cv2.putText(imgcv, 'gong count: '+str(len(ids)), (10,70),0, 1e-3 * h, (0,0,255),thick//6)

        lineThickness = 2
        cv2.line(imgcv, (llx1, lly1), (llx2, lly2), (0,255,0), lineThickness)

	return imgcv
