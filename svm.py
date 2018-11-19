import numpy as np
from sklearn import svm
import random
import json
from matplotlib.image import imread;
import matplotlib.pyplot as plt;
import skimage.transform


# HELPER
def getTransformedMatrix(image):
    target_size = 224
    number_colors = 3
    num_rows = image.shape[0]
    num_columns = image.shape[1]
    num_colors = image.shape[2]
    num_pixels =  num_rows*num_columns
    # scaled = image.reshape(num_pixels, num_colors)
    resized = skimage.transform.resize(image, (target_size,target_size, number_colors))
    resizedReshaped = resized.reshape(1, target_size * target_size * number_colors)
    return resizedReshaped

def printShape(image, name):
    print("BEFORE: ", name, "= " , image.shape) # (128, 128, 3)
    rgb_matrix =  getTransformedMatrix(image).shape # (16384, 3)
    print("AFTER: ", name, "= " , rgb_matrix)

# RANDOM
random.seed(229)

# PATH
local_path = ''
sage_path = '/home/ec2-user/SageMaker/efs/amazon-bin/'
env_path = local_path

train_file = env_path+'counting_train.json'
# train_file = env_path+'/input/counting_train.json'

image_path = env_path+'data/Images/'
# img_path = env_path+'/data/bin-images-resize/'


# FILE
with open(train_file) as f:
    data_list = json.loads(f.read())
print("#files_train=", len(data_list))

sample_images = "data/Images/"
sample_meta = "data/Metadata/"
sample_name = "00001"

sample_image = imread(sample_images+sample_name+".jpg")
sample_transformed = getTransformedMatrix(sample_image)

# CROSS VALIDATION
def getXY(setName):
    X_set = []
    Y_out = []
    bad_count = 0
    train_xId_y_list = env_path+setName+".json"
    with open(train_xId_y_list) as metadata_file:
        metadata_json = json.load(metadata_file)
    for xId_y in metadata_json:
        # print("xId_y=",xId_y)
        file_name = '%05d.jpg' % (xId_y[0]+1)
        expected_quantity = xId_y[1]
        try:
            this_image = imread(sample_images+file_name)
            image_transformed = getTransformedMatrix(sample_image)
            if len(X_set)==0:
                X_set = image_transformed
                Y_out = [expected_quantity]
            else:
                X_set = np.concatenate((X_set, image_transformed))
                Y_out = np.concatenate((Y_out, [expected_quantity]))
            print(setName + " X_set=", X_set.shape, " Y_out=", Y_out.shape)
        except:
            bad_count = bad_count+1
            # print("error=", file_name)
    return X_set,Y_out

X_train,Y_train=getXY("counting_train")
print("X_train=", X_train.shape, " Y_out=", Y_train.shape)

X_validation,Y_validation=getXY("counting_val")
print("X_validation=", X_validation.shape, " Y_out=", Y_validation.shape)


# plt.imshow(sample_image)
# plt.show()


# SVM
print("WILL NOW TRAIN SVM... set_size=", len(Y_train))
clf = svm.SVC(gamma='scale')
clf.fit(X_train, Y_train)


# ACCURACY
def getAccuracy(X_set, Y_set, class_id):
    class_total_count = 0
    class_success_count = 0
    for i in range(len(Y_set)):
        x_input = X_set[i]
        y_actual_output = Y_set[i]
        if(y_actual_output==class_id):
            y_predicted_output = clf.predict([x_input])
            # print("cross-validation predict=", y_predicted_output, " vs=", y_actual_output)
            if(y_actual_output==y_predicted_output):
                class_success_count = class_success_count + 1
            class_total_count = class_total_count+1
    class_accuracy = 0
    try:
        class_accuracy = class_success_count/class_total_count
    except:
        accuracy_error = True

    return class_accuracy, class_success_count, class_total_count

print("WILL NOW VALIDATE SVM... set_size=", len(Y_validation))
classes_under_study = 5
class_accuracy_percent = np.zeros(classes_under_study)
class_success_count = np.zeros(classes_under_study)
class_count = np.zeros(classes_under_study)

for class_id in range(1, classes_under_study+1):
    class_id_accuracy, class_id_success_count, class_id_count = getAccuracy(X_validation, Y_validation, class_id)
    print("class_id=", class_id, " class_id_accuracy=",class_id_accuracy, " class_id_success_count=",class_id_success_count, " class_id_count=",class_id_count)
    class_accuracy_percent[class_id-1] = class_id_accuracy
    class_success_count[class_id-1] = class_id_success_count
    class_count[class_id-1] = class_id_count

print("validation total=", len(Y_validation), " split into class_count=", class_count)
print("validation class_accuracy=", class_accuracy_percent)
overall_accuracy = np.sum(class_success_count)/len(Y_validation)
print("overall accuracy=", overall_accuracy)