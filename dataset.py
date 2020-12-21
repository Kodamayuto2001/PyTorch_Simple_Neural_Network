import torchvision
import numpy as np
import torch 
import cv2 
import os 

class Dataset(torch.utils.data.Dataset):
    def __init__(self,test_dataset="test-dataset/ando/",label=0,imgSize=160):
        self.data       =   self.__kobetu_dataset(test_dataset=test_dataset,imgSize=imgSize)
        self.label      =   label

    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        out_data    =   self.data
        out_label   =   self.label

        # if self.transform:
        #     out_data    =   self.transform(out_data)

        return out_data,out_label 

    
    def __kobetu_dataset(self,test_dataset,imgSize):
        f_nameList      =   os.listdir(test_dataset)
        self.data_len   =   len(f_nameList)

        img_list    =   []
        for fname in f_nameList:
            path    =   test_dataset    +   fname
            img     =   cv2.imread(path)
            #   グレースケール
            img     =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img     =   cv2.resize(img,(imgSize,imgSize)) 
            img     =   np.reshape(img,(1,imgSize,imgSize))
            # img     =   np.transpose(img,(2,0,1))
            #   正規化
            # img     =   torchvision.transforms.ToTensor()(img)
            
            img_list.append(img)
        
        img_ndarray =   np.array(img_list)
        # img_tensor  =   torchvision.transforms.ToTensor()(img_ndarray)
        return img_ndarray 