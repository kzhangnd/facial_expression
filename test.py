import cv2
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
import time
import sys
from tqdm import tqdm
from os import path
from sklearn import metrics


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


model_save_path = "model/epoch46_acc0.8703.pth" # change this to the path of the model
result_save_path = "scn_predictions.txt"

# loading 

preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
res18 = Res18Feature(pretrained = False)
checkpoint = torch.load(model_save_path)
res18.load_state_dict(checkpoint['model_state_dict'])
res18.cuda()
res18.eval()

test_list = pd.read_csv(sys.argv[1], delimiter=" ", header=None)
test_list = np.asarray(test_list)

p_path = "aligned/"
image_list = []
labels = []
for line in tqdm(test_list):
    # we only want testing data
    if "train" in line[0]:
        continue

    new_path = path.join(p_path, line[0])[:-4] + "_aligned.jpg"
    image_list.append(new_path)
    labels.append(line[1]-1) # start from 0


if path.exists(result_save_path): # if results precalculated
    predictions = pd.read_csv(result_save_path, delimiter=" ", header=None)
    predictions = np.asarray(predictions)

else:    
    predictions = []

    for img_path in tqdm(image_list):
        image = cv2.imread(img_path)
        image = image[:, :, ::-1] # BGR to RGB
        image_tensor = preprocess_transform(image)
        #print(image_tensor.shape)
        tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)

        #print(tensor.shape) #[1,3, 224, 224]
        tensor=tensor.cuda()
        #print(tensor.shape)

        time2=time.time()
        _, outputs = res18(tensor)
        _, predicts = torch.max(outputs, 1)
        #print(outputs)
        predicts = predicts[0].cpu().numpy()

        print(predicts)
        predictions.append(predicts)

    np.savetxt(result_save_path, predictions, delimiter=' ', fmt='%s')

print("SCN:", metrics.accuracy_score(labels, predictions))


