import numpy as np
from sklearn import svm
import random
import json
from matplotlib.image import imread;
import matplotlib.pyplot as plt;
import skimage.transform


## CONFIG
max_taining = 400

# PATH
# local
local_root = ''
local_images_path = local_root+'data/Images/'
local_metadata_path = local_root+'data/Metadata/'
local_summary_path = local_root+''
# sage
sage_root = '/home/ec2-user/SageMaker/efs/'
sage_images_path = sage_root+'amazon-bin/bin-images/'
sage_metadata_path = sage_root+'amazon-bin/metadata/'
sage_summary_path = sage_root
# env
env_images_path = local_images_path
env_metadata_path = local_metadata_path
env_summary_path = local_summary_path
# env_images_path = sage_images_path
# env_metadata_path = sage_metadata_path
# env_summary_path = sage_summary_path

# HELPER
def getResizedReshapedMatrix(image):
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

def getZeroMeanNormalizedVarianceMatrix(Xmatrix):
    # Mean subtraction: subtracting the mean across every individual feature in the data
    Xmatrix -= np.mean(Xmatrix, axis = 0)
    print("XmatrixMean=",Xmatrix)
    # Normalization, two wasy:
    # 1) is to divide each dimension by its standard deviation, once it has been zero-centered
    Xmatrix /= np.std(Xmatrix, axis = 0)
    print("XmatrixVariance=", Xmatrix)
    # 2) Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively
    return Xmatrix

# RANDOM
random.seed(229)


# CROSS VALIDATION
def getXY(setName):
    X_set = []
    Y_out = []
    total_count = 0
    bad_count = 0
    train_xId_y_list = env_summary_path+setName+".json"
    with open(train_xId_y_list) as metadata_file:
        metadata_json = json.load(metadata_file)
    for xId_y in metadata_json:
        if(total_count<max_taining):
            # print("xId_y=",xId_y)
            file_name = '%05d.jpg' % (xId_y[0]+1)
            expected_quantity = xId_y[1]
            try:
                this_image = imread(env_images_path+file_name)
                image_resized_reshaped = getResizedReshapedMatrix(this_image)
                image_to_use = image_resized_reshaped
                if len(X_set)==0:
                    X_set = image_to_use
                    Y_out = [expected_quantity]
                else:
                    X_set = np.concatenate((X_set, image_to_use))
                    Y_out = np.concatenate((Y_out, [expected_quantity]))
                print(setName + " X_set=", X_set.shape, " Y_out=", Y_out.shape)
            except:
                bad_count = bad_count+1
                # print("error=", file_name)
            total_count = total_count+1
    return X_set,Y_out

X_train,Y_train=getXY("counting_train")
X_validation,Y_validation=getXY("counting_val")

X_train_mean_variance_normalized = getZeroMeanNormalizedVarianceMatrix(X_train)
X_validation_mean_variance_normalized = getZeroMeanNormalizedVarianceMatrix(X_validation)

print ("X_train=", X_train_mean_variance_normalized)
# exit(1)

# PLOT
# sample_name = "00001"
# sample_image = imread(images_path+sample_name+".jpg")
# plt.imshow(sample_image)
# plt.show()


# SVM
# SVC and NuSVC are similar methods, but accept slightly different sets of parameters and have different mathematical
# formulations (see section Mathematical formulation). On the other hand, LinearSVC is another implementation of
# Support Vector Classification for the case of a linear kernel. Note that LinearSVC does not accept keyword kernel,
# as this is assumed to be linear.
# isProbability = True
print("WILL NOW TRAIN SVM... set_size=", len(Y_train))
# clf = svm.LinearSVC(loss='l2', penalty='l1', dual=False)
# clf = svm.LinearSVC(penalty='l2')
# clf = svm.NuSVC()
# clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
# clf = svm.SVC(gamma='scale')
clf = svm.NuSVC(nu=0.1) # gamma='scale'
# clf = svm.LinearSVC(penalty='l2', multi_class='ovr')
clf.fit(X_train_mean_variance_normalized, Y_train)


# ACCURACY
def getAccuracy(X_set, Y_set, class_id):
    class_total_count = 0
    class_success_count = 0
    for i in range(len(Y_set)):
        x_input = X_set[i]
        y_actual_output = Y_set[i]
        if(y_actual_output==class_id):
            y_predicted_output = clf.predict([x_input])
            print("cross-validation predict=", y_predicted_output, " vs=", y_actual_output)
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
classes_under_study = 6
class_accuracy_percent = np.zeros(classes_under_study)
class_success_count = np.zeros(classes_under_study)
class_count = np.zeros(classes_under_study)

for class_id in range(classes_under_study):
    class_id_accuracy, class_id_success_count, class_id_count = getAccuracy(X_validation_mean_variance_normalized, Y_validation, class_id)
    print("class_id=", class_id, " class_id_accuracy=",class_id_accuracy, " class_id_success_count=",class_id_success_count, " class_id_count=",class_id_count)
    class_accuracy_percent[class_id] = class_id_accuracy
    class_success_count[class_id] = class_id_success_count
    class_count[class_id] = class_id_count

print("validation total=", len(Y_validation), " split into class_count=", class_count)
print("validation class_accuracy=", class_accuracy_percent)
overall_accuracy = np.sum(class_success_count)/len(Y_validation)
print("overall accuracy=", overall_accuracy)