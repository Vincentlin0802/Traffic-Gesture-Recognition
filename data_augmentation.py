import numpy as np
import pickle


class DataAugmenter:
    def __init__(self,train_skeleton_path, file_name, augment_ratio):
        self.train_skeleton = self.load_skeleton(train_skeleton_path)
        self.file_name = file_name
        self.augment_ratio = augment_ratio

    def load_skeleton(self, skeleton_path):
        with open(skeleton_path, 'rb') as file:
            return pickle.load(file)
        
    def add_noise(self, sequence, noise_level):
        sequence = np.array(sequence)
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def augment(self, sequence):
        choice = np.random.choice(['add_noise','none'])
        if choice == 'add_noise':
            noise_level = np.random.uniform(0.02, 0.1)
            return self.add_noise(sequence, noise_level)
        else:
            return sequence
        
    def augment_dataset(self, sequences, augment_ratio=0.3):
        num_to_augment = int(len(sequences) * augment_ratio)
        indices_to_augment = np.random.choice(len(sequences), num_to_augment, replace=False)
        for idx in indices_to_augment:
            sequences[idx] = self.augment(sequences[idx])
        return sequences

    def save_augment_skeleton(self):
        augment_data = self.augment_dataset(self.train_skeleton,self.augment_ratio)
        with open(self.file_name, 'wb') as file:
            pickle.dump(augment_data, file)