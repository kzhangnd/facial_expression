import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os
from os import path
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn import metrics
import pickle
    
def data_preprocess(feature_path, label_path, test_feature_path, test_label_path):

    if not path.exists(feature_path):
        sys.exit(f'{feature_path} does not exist')

    X = np.load(feature_path) # read feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # save the scaler
    filename = 'scaler.pkl'
    pickle.dump(scaler, open(filename, 'wb'))

    if not path.exists(label_path):
        sys.exit(f'{label_path} does not exist')

    y = np.asarray(pd.read_csv(label_path, delimiter=" ", header=None)).squeeze() # read label

    X_test = np.load(test_feature_path)
    X_test = scaler.transform(X_test)
    y_test = np.asarray(pd.read_csv(test_label_path, delimiter=" ", header=None)).squeeze() # read label

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=42)

    X_train = X
    y_train = y

    print(len(X_train))
    print(len(X_test))



    clf_linear = SVC(kernel='linear')
    a = time.time()
    clf_linear.fit(X_train, y_train)
    b = time.time()
    print(f'Linear SVM used {b-a} seconds')

    # save model
    filename = 'svm_linear.pkl'
    pickle.dump(clf_linear, open(filename, 'wb'))

    # get accuracy
    result = clf_linear.predict(X_test)
    print("- Linear SVM:", metrics.accuracy_score(y_test, result))
    
    '''
    clf_poly = SVC(kernel='poly')
    a = time.time()
    clf_poly.fit(X_train, y_train)
    b = time.time()
    print(f'Poly SVM used {b-a} seconds')
    print("- Poly SVM:", metrics.accuracy_score(y_test, clf_poly.predict(X_test)))
    

    '''
    
    #clf_forest = RandomForestClassifier()
    #clf_mlp = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=1000, tol=0.001, random_state=42)



    #a = time.time()
    #scores = cross_val_score(clf_forest, X, y, cv=10)
    #b = time.time()
    #print(f'Random Forest used {b-a} seconds')
    #print(f'Accuracy: {np.mean(scores)}')

    #a = time.time()
    #scores = cross_val_score(clf_mlp, X, y, cv=10)
    #b = time.time()
    #print(f'MLP used {b-a} seconds')
    #print(f'Accuracy: {np.mean(scores)}')
    
    '''
    print("Accuracy:")
    print("- KNN:", metrics.accuracy_score(y_test, clf_knn.predict(X_test)))
    print("- ID3:", metrics.accuracy_score(y_test, clf_id3.predict(X_test)))
    print("- ID3 (overfitting):", metrics.accuracy_score(y_test, clf_id3_overfit.predict(X_test)))
    print("- CART:", metrics.accuracy_score(y_test, clf_cart.predict(X_test)))
    print("- Naive Bayes:", metrics.accuracy_score(y_test, clf_bayes.predict(X_test)))
    print("- RBF Kernel SVC:", metrics.accuracy_score(y_test, clf_rbf.predict(X_test)))
    print("- Random Forest:", metrics.accuracy_score(y_test, clf_forest.predict(X_test)))
    print("- AdaBoost:", metrics.accuracy_score(y_test, clf_boost.predict(X_test)))
    print("- MLP:", metrics.accuracy_score(y_test, clf_mlp.predict(X_test)))
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feature", required=True,
        help="path to feature file")
    parser.add_argument("-l", "--label", required=True,
        help="path to lable file")
    parser.add_argument("-tf", "--t_feature", required=True,
        help="path to test feature file")
    parser.add_argument("-tl", "--t_label", required=True,
        help="path to test lable file")
    args = parser.parse_args()
    
    data_preprocess(args.feature, args.label, args.t_feature, args.t_label)