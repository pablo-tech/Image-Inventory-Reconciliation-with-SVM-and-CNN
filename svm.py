from sklearn import svm
import random
import json


random.seed(229)

local_path = ''
sage_path = '/home/ec2-user/SageMaker/efs/amazon-bin/'
env_path = local_path

train_file = env_path+'counting_train.json'
# train_file = env_path+'/input/counting_train.json'

image_path = env_path+'data/Images/'
# img_path = env_path+'/data/bin-images-resize/'


with open(train_file) as f:
    data_list = json.loads(f.read())
print("#files_train=", len(data_list))


X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC(gamma='scale')


clf.fit(X, y)

print("predict=", clf.predict([[2., 2.]]))