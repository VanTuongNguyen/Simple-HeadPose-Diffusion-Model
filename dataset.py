from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class ZLDataset(Dataset):
    def __init__(self, data_path, transform = None):
        self.data_path = data_path
        self.data_df = pd.read_csv(self.data_path + "info.csv") 
        self.transform = transform
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
    def __getitem__(self, idx):
        data = self.data_df.iloc[idx]
        image = cv2.imread(self.data_path + "/images/" + data[4])
        id = data[0]
        text = " ".join(data[1:4])
        encoded_captions = tokenizer(
            text, padding=True, truncation=True, max_length=200
        )
        if self.transform:
            image = self.transform(image=image)["image"]
        return {"image" = image, "id"= id, "encoded_captions" = encoded_captions}
        
    def __len__(self):
        return len(self.data_df)

class HeadposeDataset_3x3_glob(Dataset):

    def __init__(self, data_path,
                transform=None):
        self.data = glob.glob(data_path+"/*/*")
        self.transform = transform
        
        x = cv2.imread(self.data[0],0)
        y = np.array(list(map(float,self.data[0].split("/")[-1][:-4].split("_"))))
        if len(y) == 3:
            y = R.from_euler('yzx', [y[0], y[1], y[2]], degrees=True).as_matrix()
        y = y.reshape((9))
        print('x (images) shape: ', x.shape)
        print('y (poses) shape: ', y.shape)

    def set_transform(self,transform):
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = cv2.imread(self.data[idx],0)
        y = np.array(list(map(float,self.data[idx].split("/")[-1][:-4].split("_"))))
        if len(y) == 3:
            y = R.from_euler('yzx', [y[0], y[1], y[2]], degrees=True).as_matrix()
        y = y.reshape((9))

        if(self.transform):
            x = self.transform(image=x)["image"]
        
        return x,y
