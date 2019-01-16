import os
import glob
import torch
import torchvision
from PIL import Image
import csv
import cv2
from numpy import array
from torchvision import datasets,transforms
from torch.utils.data import Dataset

def Transform():
    return transforms.Compose([transforms.ToTensor()])

class datamanager(Dataset):
    def __init__(self,root="E:\\Datasets\\MURA-v1.1\\train_labeled_studies.csv",key_str="SHOULDER",transform=Transform()):
        images=[]
        labels=[]
        fileContent=csv.reader(open(root,"r"))
        self.flag = 0
        for path,label in fileContent:
            if key_str in path:
                image_local=glob.glob(os.path.join("E:/Datasets/",path,"*.png"))
                self.flag=1
                #label=int(label)
                num_local=len(image_local)
                #print(label)
                for image in image_local:
                    images.append(image)
                    labels.append(label)

        self.images=images
        self.labels=labels
        self.transform=transform
        self.LabelTransform=transforms.ToTensor()
        print("self.flag{}".format(self.flag))

    def __getitem__(self,index):
        image_path=self.images[index]
        label=self.labels[index]
        image=cv2.imread(image_path)
        #image=Image.open(image_path)#.convert('RGB')
        #print(image.mode)
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        ###print(image.mode)
        if self.transform:
            #print(type(label))
            label=int(label)
            image=self.transform(image)

        #print("successful2")
        return image,label

    def __len__(self):
        return len(self.images)

'''''
images=datamanager(root="E:\\Datasets\\MURA-v1.1\\train_labeled_studies.csv",transform=Transform)
#print(images.labels)
for image,label in images:
    print(image.shape)
    print(torch.is_tensor(image))
    print(label)
'''''

