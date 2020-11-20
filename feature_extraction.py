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

def feature_extraction(image_list, root_path):
    # initialize face_alignment's face landmark predictor
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    features = []
    remove = []
    allert = []
    print(f'Extracting landmarks ...')
    for image_label in tqdm(image_list):
        image_path = path.join(root_path, image_label)[:-4] + '_aligned.jpg'
        if not path.isfile(image_path):
            sys.exit(f'The image {image_path} is not legal!')

        if 'test' in image_label:
            continue

        # load the input image, resize it, and convert it to grayscale
        image = io.imread(image_path)
        preds = fa.get_landmarks(image)
        if preds == None:
            print(f'There is no face in {image_path}')
            remove.append(image_label)
            continue

        if len(preds) != 1:
            print(f'There are {len(preds)} faces in {image_path}')
            allert.append(image_label)
            continue

        features.append(preds[0])


    np.savetxt('allert_train.txt', allert, delimiter=' ', fmt='%s')
    np.savetxt('remove_train.txt', remove, delimiter=' ', fmt='%s')
    return features
        
    
def data_preprocess(image_list_path, root_path):
    image_list = pd.read_csv(image_list_path, delimiter=" ", header=None)
    image_list = np.asarray(image_list)

    feature_file_path = 'test_2d.npy'
    if not path.exists(feature_file_path): # if feature not calculated
        X = feature_extraction(image_list[:, 0], root_path, shape_predictor)
    else:
        X = np.load(feature_file_path)
    
    remove = set(list(np.asarray(pd.read_csv('remove_test.txt', delimiter=" ", header=None)).squeeze()))
    allert = set(list(np.asarray(pd.read_csv('allert_test.txt', delimiter=" ", header=None)).squeeze()))

    y = []
    for x in tqdm(image_list):
        if 'test' not in x[0]: # only training dataset
            continue
        if x[0] in remove or x[0] in allert: # we don't want the one without face or multiple faces detected
            continue
        y.append(x[1])

    np.savetxt('test_labels.txt', y, delimiter=' ', fmt='%s')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_list_path", required=True,
        help="path to train and test image list with label")
    parser.add_argument("-r", "--root_path",
        help="root path to the images", default='aligned/')
    args = parser.parse_args()
    
    data_preprocess(args.image_list_path, args.root_path)