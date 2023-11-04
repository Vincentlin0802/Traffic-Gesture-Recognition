import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from RNN_model import *
from train_test_function import *
from get_skeleton import *
from RNN_data import *
from data_augmentation import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

data_path = './gesture_data'
train_frame_path = './image_skeleton_data/train_frames_data.pkl'
test_frame_path = './image_skeleton_data/test_frames_data.pkl'
train_skeleton_path ='./image_skeleton_data/train_skeleton_data.pkl'
test_skeleton_path ='./image_skeleton_data/test_skeleton_data.pkl'
train_skeleton_aument_path ='./image_skeleton_data/train_skeleton_aument.pkl'
save_model_path = './rnn_model'

# RNN architecture
RNN_hidden_layers = 2
RNN_hidden_nodes = 72
RNN_FC_dim = 64
dropout_p = 0.1 

# training parameterss
num_classes = 9        # number of target categorys
num_head = 8            # number of multi-head Attention Mechanism
epochs = 150      # training epochs
batch_size = 256
learning_rate = 1e-3

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

train_list, test_list, train_label, test_label = SkeletonDataset.preprocess_data(data_path)

#get image weight,height, RGB information from train and test list
train_image = ImageProcessor(data_path, train_list)
test_image = ImageProcessor(data_path, test_list)

# check if file exist or not, if not exist, run these code
if not os.path.exists(train_frame_path):
    train_image.save_image_frames(train_frame_path)
if not os.path.exists(test_frame_path):
    test_image.save_image_frames(test_frame_path)

train_frames = train_image.load_image_frames(train_frame_path)
test_frames = test_image.load_image_frames(test_frame_path)

#process skeleton data from train and test image frames
train_skeleton = SkeletonProcessor(train_frames)
test_skeleton = SkeletonProcessor(test_frames)
if not os.path.exists(train_skeleton_path):
    train_skeleton.save_image_skeleton(train_skeleton_path)
if not os.path.exists(test_skeleton_path):
    test_skeleton.save_image_skeleton(test_skeleton_path)


augmenter = DataAugmenter(train_skeleton_path,train_skeleton_aument_path,augment_ratio=0.3)
if not os.path.exists(train_skeleton_aument_path):
    augmented_sequence = augmenter.save_augment_skeleton()


skeleton_dataset = SkeletonDataset(train_skeleton_aument_path, test_skeleton_path, train_label, test_label, batch_size=10)


# set and initialize RNN model parameters for training
rnn_skeleton = RNN(RNN_embed_dim=99, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes= num_classes,num_heads= num_head ).to(device)

epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []
all_predout = []
all_testout = []
rnn_params = list(rnn_skeleton.parameters())
optimizer = torch.optim.Adam(rnn_params, lr=learning_rate)
loss_value = float('inf')
epoch_value = 0

#initialize RNN trainer
trainer = RNNTrainer(
    model=rnn_skeleton, 
    device=device, 
    train_loader=skeleton_dataset.train_loader, 
    test_loader=skeleton_dataset.test_loader, 
    optimizer=optimizer, 
    save_model_path=save_model_path,
    total_epochs = epochs
)


for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = trainer.train(epoch)
    epoch_test_loss, epoch_test_score, best_loss, best_epoch, current_testout, current_predout = trainer.test(epoch,loss_value, epoch_value)
    loss_value = best_loss
    epoch_value = best_epoch
    current_predout = current_predout.tolist()
    current_testout = current_testout.tolist()
    all_predout.append(current_predout)
    all_testout.append(current_testout)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)

final_predout = [item for sublist in all_predout for item in sublist]
flat_predout = [item for sublist in final_predout for item in sublist]
final_testout = [item for sublist in all_testout for item in sublist]
#flat_testout = [item for sublist in final_testout for item in sublist]


# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = os.path.join(save_model_path, "fig_RNN_train.png")
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()

labelname = [
    "NO GESTURE",
    "STOP",
    "MOVE STRAIGHT",
    "LEFT TURN",
    "LEFT TURN WAITING",
    "RIGHT TURN",
    "LANE CHANGING",
    "SLOW DOWN",
    "PULL OVER"
]
report = classification_report(final_testout, final_predout, target_names=labelname, digits=4,output_dict=True)
report_visualize = pd.DataFrame(report).transpose()
print(report_visualize)
report_visualize  = report_visualize.drop(['accuracy', 'macro avg', 'weighted avg'])
plt.figure(figsize=(10, 5))
sns.heatmap(report_visualize[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues')
plt.title('Classification Report Heatmap')
title = os.path.join(save_model_path, "fig_RNN_classification_report.png")
plt.savefig(title, dpi=300, bbox_inches='tight')
plt.show()
print("best loss: ", best_loss)
print("best epoch:",best_epoch+1)
