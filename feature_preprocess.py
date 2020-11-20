from tqdm import tqdm
import numpy as np
import sys
import os
import math
from os import path
import argparse

# ideas from: https://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
def convert(feature):
    xlist = []
    ylist = []
    for (x, y) in feature: # Store X and Y coordinates in two lists
        xlist.append(x)
        ylist.append(y)

    xmean = np.mean(xlist) # Find both coordinates of centre of gravity
    ymean = np.mean(ylist)

    xcentral = [(x-xmean) for x in xlist] # Calculate distance centre <-> other points in both axes
    ycentral = [(y-ymean) for y in ylist]

    #Calculate the trait norm (point 28, nose top)
    trait_norm = np.linalg.norm(np.asarray((ylist[27], xlist[27]) - np.asarray(ymean, xmean)))

    
    ''' method proposed by the blog
    landmarks_vectorised = []
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        landmarks_vectorised.append(w)
        landmarks_vectorised.append(z)

        meannp = np.asarray((ymean,xmean))
        coornp = np.asarray((z,w))
        dist = np.linalg.norm(coornp-meannp)
        landmarks_vectorised.append(dist) # norm of the vector
        landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi)) # angle of the vector
    '''

    landmarks_vectorised = []
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        landmarks_vectorised.append(x/trait_norm)
        landmarks_vectorised.append(y/trait_norm)
    
    
    return landmarks_vectorised
    

def feature_preprocess(feature_path):

    if not path.exists(feature_path): # if feature not calculated
        sys.exit(f'{feature_path} does not exist!')
    
    X = np.load(feature_path)
    
    result = []
    for feature in tqdm(X):
        result.append(convert(feature)) # convert each feature

    file_name = 'trait_' + path.split(feature_path)[1]
    np.save(file_name, np.asarray(result))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature", required=True,
        help="path to the feature file")
    args = parser.parse_args()
    
    feature_preprocess(args.feature)