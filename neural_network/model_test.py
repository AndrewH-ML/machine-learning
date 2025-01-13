import numpy as np
import pickle 
import matplotlib.pyplot as plt
import time
import glob as glob
from PIL import Image
import json
from neural_network import model_2_optimized, predict, model_3, search_params
from optimization_methods import *
from sklearn.model_selection import train_test_split
import os 


path = "/home/andrewh/tumor-prediction/data/mri/"

# im = np.array(Image.open(os.path.join(path, "yes/y0.jpg")).resize((200,200)))
# plt.imshow(im)
# plt.title("Tumor mri")
# plt.show()
# im = im/255.
# im.shape

X = []
Y = []

for file in glob.glob(os.path.join(path, 'no/*.jpg')):
  im = np.array(Image.open(file).resize((200, 200)).convert('RGB'))
  X.append(im)
  Y.append(0)

for file in glob.glob(os.path.join(path, 'yes/*.jpg')):
  im = np.array(Image.open(file).resize((200, 200)).convert('RGB'))
  X.append(im)
  Y.append(1)

X = np.array(X)
Y = np.array(Y)

print("image data processed...")
print("X shape: ", X.shape)
print("Y shape: ", Y.shape, "\n")

#transform data
print("flattening data...")
X = X.reshape(X.shape[0], -1)
Y = Y.reshape(Y.shape[0], 1)

print("data flattened...")
print("X shape: ", X.shape)
print("Y shape: ", Y.shape, "\n")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape, "\n")

print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape, "\n")

print("Transposing train and test sets...")
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T
print("new shape of train and test sets: \n")
print("\t X_train shape: ", X_train.shape)
print("\t Y_train shape: ", Y_train.shape, "\n")

print("\t X_test shape: ", X_test.shape)
print("\t Y_test shape: ", Y_test.shape, "\n")

#standardize data
print("Standardizing data... \n")
X_train, X_test = X_train/255., X_test/255.
print("Data Standardized...")

np.savez('train_test.npz', X_train=X_train, Y_train=Y_train, 
         X_test=X_test, Y_test=Y_test)
 
layers_dims = [X_train.shape[0], 20, 7, 5, 3, 1] #  4-layer model

optimizers = ['momentum', 'adam']
lr_0s = [.001, .01, .1]
decay_rates = [.001, .01, .1]
lambds = [0]

mini_batch_size = 64
beta = 0.9 # default model 3 value
beta1 = 0.9 # default 
beta2 = 0.999 # default 
epsilon = 1e-8 # default
num_epochs = 40
print_cost = False
decay = update_lr


params = [layers_dims, mini_batch_size, beta, beta1, beta2, epsilon, 
          num_epochs, print_cost, decay]

scores = search_params(model_3, X_train, X_test, Y_train, Y_test, params, optimizers, lr_0s, decay_rates, lambds)

with open('scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print("Scores saved to scores.json")
