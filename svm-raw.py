import numpy as np
from sklearn import svm
import random
import json
from matplotlib.image import imread;
import matplotlib.pyplot as plt;
import skimage.transform
import sys
from sklearn.decomposition import PCA

###
# -1) 2 digit deecimal
# 0) accuracy against train set: high/low -> overfit/bias
# 1) x-validate: maximize use of train set
# 2) param search in validation set: C, gamma
# 3) xgradient: Sravan

## CONFIG
isLocal = True # false for SageMaker
isPreproces = False # false to build model from pre-processed file
isExplore = True # true for parameter search, false for explotaition of good params
max_taining_examples = 35
if(isExplore==False):
    max_taining_examples = max_taining_examples*3
example_batch_size = 5
pca_columns = int(example_batch_size*1)

# PATH
# local
local_root = ''
local_images_path = local_root+'data/Images/'
local_metadata_path = local_root+'data/Metadata/'
local_processed_path = local_root+'data/Processed/'
local_summary_path = local_root+''
# sage
sage_root = '/home/ec2-user/SageMaker/efs/'
sage_images_path = sage_root+'amazon-bin/bin-images/'
sage_metadata_path = sage_root+'amazon-bin/metadata/'
sage_processed_path = sage_root+'processed/'
sage_summary_path = sage_root
# env
if(isLocal):
    env_images_path = local_images_path
    env_metadata_path = local_metadata_path
    env_processed_path = local_processed_path
    env_summary_path = local_summary_path
else:
    env_images_path = sage_images_path
    env_metadata_path = sage_metadata_path
    env_processed_path = sage_processed_path
    env_summary_path = sage_summary_path

# HELPER
def getRoundedResizedReshapedMatrix(image):
    target_size = 224
    number_colors = 3
    num_rows = image.shape[0]
    num_columns = image.shape[1]
    num_colors = image.shape[2]
    num_pixels =  num_rows*num_columns
    # scaled = image.reshape(num_pixels, num_colors)
    resized = skimage.transform.resize(image, (target_size,target_size, number_colors))
    resizedReshaped = resized.reshape(1, target_size * target_size * number_colors)
    roundedResizedReshaped = np.around(resizedReshaped, decimals=2)
    return roundedResizedReshaped

def getRoundedZeroMeanNormalizedVarianceMatrix(Xmatrix):
    # Mean subtraction: subtracting the mean across every individual feature in the data
    Xmatrix -= np.mean(Xmatrix, axis = 0)
    # print("XmatrixMean=",Xmatrix)
    # Normalization, two wasy:
    # 1) is to divide each dimension by its standard deviation, once it has been zero-centered
    # 2) Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively
    Xmatrix /= np.std(Xmatrix, axis = 0)
    # print("XmatrixVariance=", Xmatrix)
    roundedResizedReshaped = np.around(Xmatrix, decimals=2)
    print("XmatrixNormalizedRounded=", roundedResizedReshaped)
    return roundedResizedReshaped

# RANDOM
random.seed(229)


# CROSS VALIDATION
def getXY(setName, i_begin, i_end):
    X_set = []
    Y_out = []
    train_xId_y_list = env_summary_path+setName+".json"
    with open(train_xId_y_list) as metadata_file:
        metadata_json = json.load(metadata_file)
    for metadata_index in range(i_begin, i_end):
        try:
            xId_y = metadata_json[metadata_index]
            print(setName+": "+ str(i_begin)+"->"+str(i_end)+" => xId_y=",xId_y)
            file_name = '%05d.jpg' % (xId_y[0]+1)
            expected_quantity = xId_y[1]
            this_image = imread(env_images_path+file_name)
            image_resized_reshaped = getRoundedResizedReshapedMatrix(this_image)
            image_to_use = image_resized_reshaped
            if len(X_set)==0:
                X_set = image_to_use
                Y_out = [expected_quantity]
            else:
                X_set = np.concatenate((X_set, image_to_use))
                Y_out = np.concatenate((Y_out, [expected_quantity]))
        except Exception:
            print("Unexpected error:", sys.exc_info())
            bad_count = True
            # print("error=", file_name)
    print(setName + " X_set=", X_set.shape, " Y_out=", Y_out.shape)
    return X_set,Y_out

def getBatchFileName(set_name, in_or_out, batch_number):
    return env_processed_path+set_name+"."+in_or_out+str(batch_number)

# DATA
number_of_batches = int(max_taining_examples/example_batch_size)
if(isPreproces):
    ## SAVE BATCH TO DISK
    for batch in range(number_of_batches):
        i_begin = batch*example_batch_size
        i_end = i_begin + example_batch_size
        print("i_begin=", i_begin, " i_end=",i_end)
        set_name = "counting_train"
        X_train_batch,Y_train_batch=getXY(set_name, i_begin, i_end)
        np.save(getBatchFileName(set_name, "X", batch), X_train_batch)
        np.save(getBatchFileName(set_name, "Y", batch), Y_train_batch)
        set_name = "counting_val"
        X_validation_batch,Y_validation_batch=getXY(set_name, i_begin, i_end)
        np.save(getBatchFileName(set_name, "X", batch), X_validation_batch)
        np.save(getBatchFileName(set_name, "Y", batch), Y_validation_batch)
    exit(0)
# else:

## RECOVER BATHES FROM DISK
def getMatrixFromFile(set_name, in_or_out):
    full_set = []
    for batch in range(number_of_batches):
        try:
            file_name = getBatchFileName(set_name, in_or_out, batch)+".npy"
            batch_set = np.load(file_name)
            print(batch_set.shape, " batch_set=",batch_set)
            if len(full_set)==0:
                full_set = batch_set
            else:
                full_set = np.concatenate((full_set, batch_set), axis=0)
            print(full_set.shape, " full_set=",full_set)
        except Exception:
            print("unable to recover ", set_name, " ", in_or_out, " batch=" + batch)
    return full_set


X_full_train_set = getMatrixFromFile("counting_train", "X")
Y_full_train_set = getMatrixFromFile("counting_train", "Y")
X_full_validation_set = getMatrixFromFile("counting_val", "X")
Y_full_validation_set = getMatrixFromFile("counting_train", "Y")

X_train_mean_variance_normalized = getRoundedZeroMeanNormalizedVarianceMatrix(X_full_train_set)
X_validation_mean_variance_normalized = getRoundedZeroMeanNormalizedVarianceMatrix(X_full_validation_set)

# PCA
def getPcaMatrix(design_matrix, pca_model):
    pca_matrix = pca_model.fit_transform(design_matrix)
    roundedPcaMatrix = np.around(pca_matrix, decimals=2)
    print("roundedPcaMatrix[0]=", roundedPcaMatrix[0])
    return roundedPcaMatrix

# n_components=10000 must be between 0 and min(n_samples, n_features)=336 with svd_solver='full'
pca = PCA(n_components= pca_columns)  # some examples may be missing (in the test set)
X_train_pca_mean_variance_normalized = getPcaMatrix(X_train_mean_variance_normalized, pca)
X_validation_pca_mean_variance_normalized = getPcaMatrix(X_validation_mean_variance_normalized, pca)

# exit(1)

# FINAL MATRIX
X_train_final = X_train_pca_mean_variance_normalized
Y_train_final = Y_full_train_set
print (X_train_final.shape, " <= X_train_final[0]=", X_train_final[0])
X_validation_final = X_validation_pca_mean_variance_normalized
Y_validation_final = Y_full_validation_set
print (X_validation_final.shape, " <= X_validation_final[0]=", X_validation_final[0])


# ACCURACY
def getAccuracy(param_string, trained_model, X_set, Y_set, class_id):
    class_total_count = 0
    class_success_count = 0
    for i in range(len(Y_set)):
        x_input = X_set[i]
        y_actual_output = Y_set[i]
        if(y_actual_output==class_id):
            y_predicted_output = trained_model.predict([x_input])
            # print(param_string, " cross-validation predict=", y_predicted_output, " vs=", y_actual_output)
            if(y_actual_output==y_predicted_output):
                class_success_count = class_success_count + 1
            class_total_count = class_total_count+1
    class_accuracy = 0
    try:
        class_accuracy = float(class_success_count/class_total_count)
    except:
        accuracy_error = True

    return class_accuracy, class_success_count, class_total_count

def validate(param_string, trained_model, X_validation, Y_validation):
    print(param_string, " WILL NOW VALIDATE SVM... set_size=", len(Y_validation))
    classes_under_study = 6
    class_accuracy_percent = np.zeros(classes_under_study)
    class_success_count = np.zeros(classes_under_study)
    class_count = np.zeros(classes_under_study)

    for class_id in range(classes_under_study):
        class_id_accuracy, class_id_success_count, class_id_count = getAccuracy(param_string, trained_model, X_validation, Y_validation, class_id)
        print(param_string, " class_id=", class_id, " class_id_accuracy=",class_id_accuracy, " class_id_success_count=",class_id_success_count, " class_id_count=",class_id_count)
        class_accuracy_percent[class_id] = class_id_accuracy
        class_success_count[class_id] = class_id_success_count
        class_count[class_id] = class_id_count

    print(param_string, " validation_total=", len(Y_validation), " split into class_count=", class_count)
    print(param_string, " validation_class_accuracy=", class_accuracy_percent)
    overall_accuracy = np.sum(class_success_count)/len(Y_validation)
    class_and_overall_accuracy = np.append(class_accuracy_percent, overall_accuracy)
    print(param_string, " class_and_overall_accuracy=", class_and_overall_accuracy)
    return class_and_overall_accuracy


# SVM
# SVC and NuSVC are similar methods, but accept slightly different sets of parameters and have different mathematical
# formulations (see section Mathematical formulation). On the other hand, LinearSVC is another implementation of
# Support Vector Classification for the case of a linear kernel. Note that LinearSVC does not accept keyword kernel,
# as this is assumed to be linear.


# PARAM SEARCH
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# Search for params: SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def accuracy_of_svc(search_case, X_train_matrix, Y_train_matrix, X_validation_matrix, Y_validation_matrix,
                    C_range, gamma_range, accuracy_map):
    for c_param in C_range:
        for gamma_param in gamma_range:
            param_string = search_case,"_c=",c_param, "_gamma=",gamma_param
            print(param_string, "...WILL NOW TRAIN SVM... set_size=", len(Y_train_matrix))
            try:
                clf = svm.SVC(C=c_param, gamma=gamma_param)  # radial kernel
                clf.fit(X_train_matrix, Y_train_matrix)
                class_accuracy_percent = validate(param_string, clf, X_validation_matrix, Y_validation_matrix)
            except Exception:
                print("Unexpected error:", sys.exc_info())
                bad = True
                class_accuracy_percent = 0
            accuracy_map[param_string] = class_accuracy_percent
            # print("SAVING_INTO", param_string, " class_accuracy_percent=", class_accuracy_percent)
    return accuracy_map

# Search for params: Nu
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
def accuracy_of_nu(search_case, X_train_matrix, Y_train_matrix, X_validation_matrix, Y_validation_matrix,
                   nu_range, accuracy_map):
    for nu_param in nu_range:
        param_string = search_case, "_nu_param=",nu_param
        print(param_string, "...WILL NOW TRAIN SVM... set_size=", len(Y_train_matrix))
        try:
            clf = svm.NuSVC(nu=nu_param)
            clf.fit(X_train_matrix, Y_train_matrix)
            class_accuracy_percent = validate(param_string, clf, X_validation_matrix, Y_validation_matrix)
        except Exception:
            print("Unexpected error:", sys.exc_info())
            bad = True
            class_accuracy_percent = 0
        accuracy_map[param_string] = class_accuracy_percent
        # print("SAVING_INTO", param_string, " class_accuracy_percent=", class_accuracy_percent)
    return accuracy_map

# Search for params: Linear
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
def accuracy_of_linear(search_case, X_train_matrix, Y_train_matrix, X_validation_matrix, Y_validation_matrix,
                       linear_penalty, linear_loss, linear_multiclass_strategy, C_range, accuracy_map):
    for penalty_param in linear_penalty:
        for loss_param in linear_loss:
            for strategy_param in linear_multiclass_strategy:
                for c_param in C_range:
                    param_string = search_case, "_penalty=",penalty_param, "_loss=",loss_param, "_multi_class=",strategy_param, "_c_param=",c_param
                    print(param_string, "...WILL NOW TRAIN SVM... set_size=", len(Y_train_matrix))
                    try:
                        clf = svm.LinearSVC(penalty=penalty_param, loss=loss_param, multi_class=strategy_param, C=c_param)
                        clf.fit(X_train_matrix, Y_train_matrix)
                        class_accuracy_percent = validate(param_string, clf, X_validation_matrix, Y_validation_matrix)
                    except Exception:
                        print("Unexpected error:", sys.exc_info())
                        bad = True
                        class_accuracy_percent = 0
                    accuracy_map[param_string] = class_accuracy_percent
                    # print("SAVING_INTO", param_string, " class_accuracy_percent=", class_accuracy_percent)
    return accuracy_map


## ACCURACY RESULTS
# SVC
def C_range(isExplore):
    if(isExplore):
        return [1, 1e2, 1e3] # 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    return [1, 1e2, 1e3] # 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
def gamma_range(isExplore):
    if(isExplore):
        return [1e-8, 1e-7, 1e-6]  # 1e-5, 1e-9, 1e-11, 1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    return [1e-8, 1e-7, 1e-6]  # 1e-5, 1e-9, 1e-11, 1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
# Nu
def nu_range(isExplore):
    if(isExplore):
        return [1e-6, 1e-5, 1e-4, 1e-3, 0.01] # 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.20, 0.25, 0.05, 0.1, 0.15
    return [1e-6, 1e-5, 1e-4, 1e-3, 0.01] # 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.20, 0.25, 0.05, 0.1, 0.15
# Linear
def linear_penalty(isExplore):
    if(isExplore):
        return ['l1', 'l2']
    return ['l1', 'l2']
def linear_loss(isExplore):
    if(isExplore):
        return ['hinge', 'squared_hinge']
    return ['hinge', 'squared_hinge']
def linear_multiclass_strategy(isExplore):
    if(isExplore):
        return ['ovr', 'crammer_singer']
    return ['ovr', 'crammer_singer']
def C_range_explore(isExplore):
    if(isExplore):
        return [1, 1e2, 1e3, 1e4]
    return [1, 1e2, 1e3, 1e4]

def determine_model_accuracy(set_name, X_train, Y_train, X_validation, Y_validation,
                             param_accuracy, isExplore):
    accuracy_of_svc(set_name, X_train, Y_train, X_validation, Y_validation,
                    C_range(isExplore), gamma_range(isExplore), param_accuracy)
    accuracy_of_nu(set_name, X_train, Y_train, X_validation, Y_validation,
                   nu_range(isExplore), param_accuracy)
    accuracy_of_linear(set_name, X_train, Y_train, X_validation, Y_validation,
                       linear_penalty(isExplore), linear_loss(isExplore), linear_multiclass_strategy(isExplore), C_range(isExplore), param_accuracy)

def determine_accuracy(param_accuracy, isExplore):
    determine_model_accuracy("trainWithTraining_validateWithValidation", X_train_final, Y_train_final, X_validation_final, Y_validation_final,
                       param_accuracy, isExplore)
    determine_model_accuracy("trainWithTraining_validateWithTraining", X_train_final, Y_train_final, X_train_final, Y_train_final,
                       param_accuracy, isExplore)

# RUN
np.set_printoptions(precision=2)
param_accuracy = {}
determine_accuracy(param_accuracy, True)
for accuracy_key in param_accuracy.keys():
    print("==>", accuracy_key, " == ", param_accuracy[accuracy_key])

## SELECTED PARAMETERS
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1, '_gamma=', 1e-06)  ==  [0.   0.   0.22 0.6  0.23 0.   0.24]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 1e-06)  ==  [0.32 0.37 0.06 0.01 0.2  0.01 0.11]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_multi_class=', 'crammer_singer', '_c_param=', 1)  ==  [0.04 0.16 0.19 0.23 0.19 0.19 0.19]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_multi_class=', 'ovr', '_c_param=', 1)  ==  [0.02 0.09 0.17 0.25 0.23 0.18 0.19]


# PLOT
# sample_name = "00001"
# sample_image = imread(images_path+sample_name+".jpg")
# plt.imshow(sample_image)
# plt.show()


# ==> ('trainWithTraining_validateWithTraining', '_c=', 1, '_gamma=', 1e-08)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1, '_gamma=', 1e-07)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1, '_gamma=', 1e-06)  ==  [0.   0.   0.   1.   0.12 0.   0.27]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1, '_gamma=', 1e-05)  ==  [0.67 0.8  0.89 0.97 0.96 0.91 0.91]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 100.0, '_gamma=', 1e-08)  ==  [0.   0.   0.01 0.94 0.3  0.   0.3 ]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 100.0, '_gamma=', 1e-07)  ==  [0.62 0.76 0.81 0.88 0.83 0.8  0.82]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 100.0, '_gamma=', 1e-06)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 100.0, '_gamma=', 1e-05)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1000.0, '_gamma=', 1e-08)  ==  [0.62 0.75 0.81 0.86 0.83 0.79 0.81]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1000.0, '_gamma=', 1e-07)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1000.0, '_gamma=', 1e-06)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 1000.0, '_gamma=', 1e-05)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 10000.0, '_gamma=', 1e-08)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 10000.0, '_gamma=', 1e-07)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 10000.0, '_gamma=', 1e-06)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_c=', 10000.0, '_gamma=', 1e-05)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.0001)  ==  [1.   0.19 0.16 0.09 0.11 0.09 0.14]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.001)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.01)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.05)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.15)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.2)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_nu_param=', 0.25)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  0
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithTraining', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
#
#
#
#
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1, '_gamma=', 1e-08)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1, '_gamma=', 1e-07)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1, '_gamma=', 1e-06)  ==  [0.   0.   0.   0.95 0.02 0.   0.24]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1, '_gamma=', 1e-05)  ==  [0.   0.05 0.23 0.49 0.17 0.06 0.22]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 100.0, '_gamma=', 1e-08)  ==  [0.   0.   0.   0.84 0.09 0.   0.23]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 100.0, '_gamma=', 1e-07)  ==  [0.05 0.09 0.26 0.3  0.17 0.15 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 100.0, '_gamma=', 1e-06)  ==  [0.   0.08 0.21 0.29 0.22 0.19 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 100.0, '_gamma=', 1e-05)  ==  [0.   0.07 0.24 0.35 0.19 0.12 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1000.0, '_gamma=', 1e-08)  ==  [0.05 0.09 0.25 0.3  0.18 0.15 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1000.0, '_gamma=', 1e-07)  ==  [0.05 0.12 0.25 0.28 0.22 0.17 0.22]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1000.0, '_gamma=', 1e-06)  ==  [0.   0.08 0.21 0.29 0.22 0.19 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 1000.0, '_gamma=', 1e-05)  ==  [0.   0.07 0.24 0.35 0.19 0.12 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 10000.0, '_gamma=', 1e-08)  ==  [0.05 0.14 0.27 0.27 0.24 0.17 0.23]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 10000.0, '_gamma=', 1e-07)  ==  [0.05 0.12 0.25 0.28 0.23 0.17 0.22]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 10000.0, '_gamma=', 1e-06)  ==  [0.   0.08 0.21 0.29 0.22 0.19 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_c=', 10000.0, '_gamma=', 1e-05)  ==  [0.   0.07 0.24 0.35 0.19 0.12 0.21]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.0001)  ==  [0.71 0.23 0.08 0.   0.02 0.   0.06]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.001)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.01)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.05)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.1)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.15)  ==  [0.   0.   0.   1.   0.   0.   0.25]
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.2)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_nu_param=', 0.25)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  0
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l1', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'ovr', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 100.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 1000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
# ==> ('trainWithTraining_validateWithValidation', '_penalty=', 'l2', '_loss=', 'squared_hinge', '_strategy=', 'crammer_singer', '_c_param=', 10000.0)  ==  [1. 1. 1. 1. 1. 1. 1.]
