import cv2
from sys import platform as sys_pf
import warnings
import face_alignment
from face_alignment.utils import *
import pickle
from feature_preprocess import convert

warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':

    # Load the video file and count number of frames
    cam = cv2.VideoCapture(0)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # load pre-trained svm and fitted scaler for feature processing
    # change the path syntax if you are running on other OS
    clf_linear = pickle.load(open("model\svm_linear.pkl", 'rb'))
    scaler = pickle.load(open("model\scaler.pkl", 'rb'))

    # label of the expression
    ex_labels = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    print("Tracking started...")

    while (True):

        ret, curr_frame = cam.read()

        preds = fa.get_landmarks(curr_frame)

        if preds == None:
            continue

        # feature processing
        converted_features = []
        for i in range(len(preds)):
            converted_features.append(convert(preds[i]))

        # normalize using the parameter of the training data
        normalized_features = scaler.transform(converted_features)

        labels = clf_linear.predict(normalized_features)

        # get bounding box
        bb = create_bounding_box(torch.tensor(preds))
        for i in range(len(preds)):
            cv2.rectangle(curr_frame, (bb[i, 0], bb[i, 1]), (bb[i, 2], bb[i, 3]), (0, 255, 0), 3)
            cv2.putText(curr_frame,
                        ex_labels[labels[i] - 1],
                        (bb[i, 0], bb[i, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA)

        cv2.imshow("Expression Classification", curr_frame)
        action = cv2.waitKey(1)
        if action & 0xFF == 27:
            break

    print("Tracking finished.")
    cv2.destroyAllWindows()
