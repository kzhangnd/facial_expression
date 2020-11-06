import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os
from os import path
import argparse
import cv2
from sklearn import preprocessing
import face_alignment
from skimage import io

def feature_extraction(image_list, root_path, shape_predictor):
    # initialize face_alignment's face landmark predictor
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    features = []
    print(f'Extracting landmarks ...')
    for image_label in tqdm(image_list):
        image_path = path.join(root_path, image_label)
        if not path.isfile(image_path):
            sys.exit(f'The image {image_path} is not legal!')

        # load the input image, resize it, and convert it to grayscale
        image = io.imread(image_path)
        preds = fa.get_landmarks(image)

        print(preds)
        break

        '''
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # detect multiple faces
        if len(rects) != 1:
            print(f'The image {image_path} has multiple faces!')
            continue
    
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to numpy
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        '''

        # normalize the shape into range [0, 100]
        shape_scaled = preprocessing.minmax_scale(preds, feature_range=(0, 100))
        features.append(shape_scaled)

    return features
        
    
def data_preprocess(image_list_path, root_path, shape_predictor):
    image_list = pd.read_csv(image_list_path, delimiter=" ", header=None)
    image_list = np.asarray(image_list)

    X = feature_extraction(image_list[:, 0], root_path, shape_predictor)

    print(X)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape-predictor", 
        help="path to facial landmark predictor", default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument("-i", "--image_list_path", required=True,
        help="path to train and test image list with label")
    parser.add_argument("-r", "--root_path",
        help="root path to the images", default='original/')
    args = parser.parse_args()
    
    X_train, y_train, X_test, y_test = data_preprocess(args.image_list_path, args.root_path, args.shape_predictor)