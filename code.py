from PIL import Image
import numpy as np
import os
import os.path
import json
import random
import skimage.io
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from skimage import feature

random.seed(229)

train_file = '/home/ec2-user/SageMaker/efs/amazon-bin/input/counting_train.json'
img_path = '/home/ec2-user/SageMaker/efs/amazon-bin/data/bin-images-resize/'

with open(train_file) as f:
          data_list = json.loads(f.read())
print(len(data_list))

# Randomly select 100,000 images for model development
#random.Random(4).shuffle(data_list)
#data_list_sample = data_list[0:100000]
#print(data_list_sample[0])

# Model using actual features of images in rgb

# Load resized image data based on the sampled data_list
# img_name = '%05d.jpg' % (data_list_sample[5][0]+1)
# img = os.path.join(img_path, img_name)
# a = skimage.io.imread(img)
    

# data_gray = color.rgb2gray(a)

#print("data array", data_gray.shape)
#plt.imshow(data_gray)

# ppc = 16
# hog_images = []
# hog_features = []

# fd,hog_image = hog(data_gray, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(1, 1),block_norm= 'L2',visualise=True)
# hog_images.append(hog_image)
# print(fd.shape)
# hog_features.append(fd)
# print(len(hog_features))
# #print(a.shape)

# plt.imshow(hog_images[0])

# # Blob detection - Laplacian of Guassian (LoG)
# blobs_log = feature.blob_log(data_gray, max_sigma=30, num_sigma=10, threshold=.1)
# print("bob log", blobs_log)

# Distribution of various counts in the training set
labels = np.empty((len(data_list), ), dtype=int)
for idx,data in enumerate(data_list):
    labels[idx] = data[1]

plt.hist(labels, bins = 6, density = False)


# Creating the training dataset of shape (m,n)
img_path = '/home/ec2-user/SageMaker/efs/amazon-bin/data/bin-images/'
training = np.empty((len(data_list), 224 * 224 * 3), dtype=np.int8)
for idx, data in enumerate(data_list):
    img_name = '%05d.jpg' % (data[0])
    img = os.path.join(img_path, img_name)
    img_data = skimage.io.imread(img)
    img_data_resize = skimage.transform.resize(img_data, (224,224,3))
    training[idx, :] = img_data_resize.reshape(1, 224 * 224 * 3)
    
#Saving training dataset to disk
np.savetxt('/home/ec2-user/SageMaker/efs/amazon-bin/input/train_all_pixel.txt', training, delimiter=',')

#data_gray = color.rgb2gray(a)