from torch.utils import data
import torch
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

class SkeletonDataset:
    def __init__(self, train_skeleton_path, test_skeleton_path, train_label, test_label, batch_size):
        self.train_skeleton = self.load_skeleton(train_skeleton_path)
        self.test_skeleton = self.load_skeleton(test_skeleton_path)

        self.train_dataset = data.TensorDataset(torch.Tensor(self.train_skeleton), torch.LongTensor(train_label))
        self.test_dataset = data.TensorDataset(torch.Tensor(self.test_skeleton), torch.LongTensor(test_label))

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def load_skeleton(self, skeleton_path):
        with open(skeleton_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def preprocess_data(data_path):
        file_list = os.listdir(data_path)
        file_list.remove(".DS_Store")   # sometimes MacOS will auto generate this file

        all_names = []
        gesture_label = []

        for file in file_list:
            # file name example is "label_0_s0"
            loc1 = file.find('l_')
            loc2 = file.find('_s')
            gesture_label.append(file[loc1+2:loc2])  # get label from file name
            all_names.append(file)

        gesture_label = list(map(int, gesture_label)) # from string to int
        all_X_list = all_names
        all_y_list = gesture_label

        train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.3, random_state=42, stratify=all_y_list)

        return train_list, test_list, train_label, test_label