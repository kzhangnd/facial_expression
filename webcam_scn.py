import cv2
from sys import platform as sys_pf
import warnings

import face_alignment
from face_alignment.utils import *

import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn


warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out


if __name__ == '__main__':

    # Load the video file and load model
    cam = cv2.VideoCapture(0)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False) # use landmark to get bounding box
    model_save_path = "model\epoch46_acc0.8703.pth" # change this to the path of the model

    # loading network model
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    res18 = Res18Feature(pretrained=False)
    checkpoint = torch.load(model_save_path)
    res18.load_state_dict(checkpoint['model_state_dict'])
    res18.cuda()
    res18.eval()

    # label of the expression
    ex_labels = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    print("Tracking started...")

    while (True):

        ret, curr_frame = cam.read()

        preds = fa.get_landmarks(curr_frame)

        # get bounding box
        bb = create_bounding_box(torch.tensor(preds))
        bb = bb.type(torch.IntTensor)
        for i in range(len(preds)):
            cv2.rectangle(curr_frame, (bb[i, 0], bb[i, 1]), (bb[i, 2], bb[i, 3]), (0, 255, 0), 3)
            image = curr_frame[bb[i,1]:bb[i,3], bb[i,0]:bb[i,2]]
            image = image[:, :, ::-1]  # BGR to RGB
            image_tensor = preprocess_transform(image)

            tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)
            tensor = tensor.cuda()

            _, outputs = res18(tensor)
            _, predicts = torch.max(outputs, 1)

            predicts = predicts[0].cpu().numpy()

            cv2.putText(curr_frame,
                        ex_labels[int(predicts)],
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
