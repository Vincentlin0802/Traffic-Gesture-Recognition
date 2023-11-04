import cv2
import os
import pickle
import mediapipe as mp
import torch
import numpy as np

class ImageProcessor:
    def __init__(self,data_path,foler_list):
        self.datapath = data_path
        self.folder_list = foler_list

    def extract_frame_number(self, filename):
        return int(filename.split("_")[1].split(".")[0]) 

    def read_image(self,datapath,folder_list):
        frames_two =[]
        frames_three = []
        for i in range (len(folder_list)):
            folder_path = os.path.join(datapath, folder_list[i])
            image_files = os.listdir(folder_path)
            if '.DS_Store' in image_files:
                image_files.remove('.DS_Store')
            image_files = sorted(image_files, key=self.extract_frame_number)
            frames = []
            for img_file in image_files:
                image_path = os.path.join(folder_path, img_file)
                frame = cv2.imread(image_path)  # read image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frames.append(image_rgb)  #one image 
            frames_two.append(frames)   # one file image info 
        frames_three.append(frames_two) # all files image info
        return frames_three
    
    def save_image_frames(self, file_name):
        frames_data = self.read_image(self.datapath,self.folder_list)
        with open(file_name, 'wb') as file:
            pickle.dump(frames_data, file)
    
    def load_image_frames(self, file_name):
        with open(file_name, 'rb') as file:
            frames_data = pickle.load(file)
        return frames_data


class SkeletonProcessor:
    def __init__(self,image_frames):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.image_frames = image_frames

    def skeleton_transform(self, image_frames):
        skeleton_data_final = []
        for i in range (len(image_frames[0])):
            frames = image_frames[0][i]
            #print(len(frames))
            skeleton_data = []
            for frame in frames:
                results = self.pose.process(frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    frame_data = []
                    for landmark in landmarks:
                        frame_data.extend([landmark.x, landmark.y, landmark.z])
                    skeleton_data.append(frame_data)
                else:
                    # add empty skeleton key points
                    skeleton_data.append([0] * 33 * 3)  # Mediapipe Pose has 33 key points，each include x, y, z three coordinates
            #set the total frame imgages to 40, make sure the size is the same before training
            while len(skeleton_data) < 40: 
                last_data = skeleton_data[-1]
                skeleton_data.append(last_data) #if less than 40 images, append the last skeleton data
            if len(skeleton_data) > 40:
                skeleton_data = skeleton_data[:40]
            skeleton_data_final.append(skeleton_data) # if more than 40, truncate the first 40
        return skeleton_data_final
    

    def save_image_skeleton(self, file_name):
        skeleton_data = self.skeleton_transform(self.image_frames)
        with open(file_name, 'wb') as file:
            pickle.dump(skeleton_data, file)
    

