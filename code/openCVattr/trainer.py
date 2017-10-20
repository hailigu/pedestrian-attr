import argparse

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import pickle as pickle

class ERFTrainer(object):
    def __init__(self, X, label_words):
        self.le = preprocessing.LabelEncoder()
        self.clf = ExtraTreesClassifier(n_estimators=100, max_depth=16,random_state=0)
        
        y = self.encode_labels(label_words)
        self.clf.fit(np.asarray(X), y)
        
    def encode_labels(self, label_words):
        self.le.fit(label_words)
        return np.array(self.le.transform(label_words), dtype=np.float32)
    
    def classify(self, X):
        label_nums = self.clf.predict(np.asarray(X))
        label_words = self.le.inverse_transform([int(x) for x in label_nums])
        return label_words
    
if __name__=='__main__':

    feature_map_file =  'feature_map.pkl'
    model_file = 'erf.pkl'

    #Load the feature map
    with open(feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)
    
    # Extract feature vectors and the labels
    label_words = [x['object_class'] for x in feature_map]
    dim_size = feature_map[0]['feature_vector'].shape[1]  
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]
    
    # Train the Extremely Random Forests classifier
    erf = ERFTrainer(X, label_words) 
    if model_file:
        with open(model_file, 'wb') as f:
            pickle.dump(erf, f)
    
    
    
    
    