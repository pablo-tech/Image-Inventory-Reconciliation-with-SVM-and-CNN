import numpy as np
from sklearn import svm
import random
import json
from matplotlib.image import imread;
import matplotlib.pyplot as plt;
import skimage.transform
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


########
########  CONFIGURE ENVIRONMENT
########
isLocal = True # false for SageMaker
isPreproces = False # false to build model from pre-processed file
isExplore = False # true for parameter search, false for explotaition of good params
# examples
max_taining_examples = 7
if(isLocal==False):
    max_taining_examples = 1000
if(isExplore==False):
    max_taining_examples = max_taining_examples*5  ### fine-tune scale
# batch
example_batch_size = 5
if(isLocal==False):
    example_batch_size = 1000
number_of_folds=10
# pca
pca_columns = int(example_batch_size*1)
# classes
num_classes_under_study = 6
# random
random.seed(229)


# PATH TO FILES
# local
local_root = ''
local_images_path = local_root+'data/Images/'
local_metadata_path = local_root+'data/Metadata/'
local_processed_path = local_root+'data/Processed/'
local_summary_path = local_root+''
# SageMaker
sage_root = '/home/ec2-user/SageMaker/efs/'
sage_images_path = sage_root+'amazon-bin/bin-images/'
sage_metadata_path = sage_root+'amazon-bin/metadata/'
sage_processed_path = sage_root+'processed/'
sage_summary_path = sage_root
# choose path based on environment
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

########
######## HELPER FUNXTIONS
########

# Resize and Reshape Matrix
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

# Normalize: zero mean, unit variance
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


# Cross-validation; build the sub-set
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


########
########  DATA: PRE-PROCESSING
########

# File names: where images are located after pre-processing
def getBatchFileName(set_name, in_or_out, batch_number):
    return env_processed_path+set_name+"."+in_or_out+str(batch_number)

# Save batch of images to disk
number_of_batches = int(max_taining_examples/example_batch_size)
if(isPreproces):
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


# Recover batch of images from disk
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


# Principal Component Analysis
def getPcaMatrix(design_matrix, pca_model):
    pca_matrix = pca_model.fit_transform(design_matrix)
    roundedPcaMatrix = np.around(pca_matrix, decimals=2)
    print("roundedPcaMatrix[0]=", roundedPcaMatrix[0])
    return roundedPcaMatrix

# Accuracy and Confusion Matrix
def validate_model(param_string, trained_model, X_validation, Y_validation):
    y_predicted_output = trained_model.predict(X_validation)
    class_and_overall_accuracy = accuracy_score(Y_validation, y_predicted_output)
    true_vs_predicted = confusion_matrix(Y_validation, y_predicted_output)
    print(param_string, "_class_and_overall_accuracy=", class_and_overall_accuracy, "_true_vs_predicted=", true_vs_predicted)
    return class_and_overall_accuracy, true_vs_predicted


########
######## SUPPORT VECTOR MACHINE: parameter search
########
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Unbalanced Classes: weight
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
# Set the parameter C of class i to class_weight[i]*C
def get_class_weights(Y_matrix, num_classes):
    class_counts = np.zeros(num_classes)
    for y_index in range(len(Y_matrix)):
        class_id = Y_matrix[y_index]
        class_counts[class_id] = class_counts[class_id] + 1
        # print("y_index=",y_index, " class_id=",class_id)
    overall_high = np.amax(class_counts)
    # print("overall_high=",overall_high)
    class_weights = {}
    for class_id in range(num_classes):
        try:
            class_weights[class_id] = overall_high/class_counts[class_id]
        except:
            print("overall_high=",overall_high, " class_counts[]", class_counts[class_id])
    print("class_counts=",class_counts, "class_weights=", class_weights)
    return class_weights

# svm.SVC: parameter search
def performance_of_svm_svc(search_case, X_train_matrix, Y_train_matrix, X_validation_matrix, Y_validation_matrix,
                    C_range, gamma_range, accuracy_map):
    for c_param in C_range:
        for gamma_param in gamma_range:
            param_string = search_case,"_c=",c_param, "_gamma=",gamma_param
            # print(param_string, "...WILL NOW TRAIN SVM... set_size=", len(Y_train_matrix))
            try:
                clf = svm.SVC(C=c_param, gamma=gamma_param, class_weight=get_class_weights(Y_train_matrix, num_classes_under_study))  # radial kernel
                clf.fit(X_train_matrix, Y_train_matrix)
                class_accuracy_percent, true_vs_predicted_confusion_matrix = validate_model(param_string, clf, X_validation_matrix, Y_validation_matrix)
                accuracy_and_confusion = "_accuracy=", str(class_accuracy_percent) + "_confusion=" + str(true_vs_predicted_confusion_matrix)
                accuracy_map[param_string] = accuracy_and_confusion
            except Exception:
                print("Unexpected error:", sys.exc_info())
                bad = True
                class_accuracy_percent = 0
    return

# svm.NuSVC: parameter search
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
def accuracy_of_nu(search_case, X_train_matrix, Y_train_matrix, X_validation_matrix, Y_validation_matrix,
                   nu_range, accuracy_map):
    for nu_param in nu_range:
        param_string = search_case, "_nu_param=",nu_param
        print(param_string, "...WILL NOW TRAIN SVM... set_size=", len(Y_train_matrix))
        try:
            clf = svm.NuSVC(nu=nu_param)
            clf.fit(X_train_matrix, Y_train_matrix)
            class_accuracy_percent, true_vs_predicted_confusion_matrix = validate_model(param_string, clf, X_validation_matrix, Y_validation_matrix)
            accuracy_and_confusion = "_accuracy=", str(class_accuracy_percent) + "_confusion=" + str(true_vs_predicted_confusion_matrix)
            accuracy_map[param_string] = accuracy_and_confusion
        except Exception:
            print("Unexpected error:", sys.exc_info())
            bad = True
            class_accuracy_percent = 0
        accuracy_map[param_string] = class_accuracy_percent
    return

# svm.Linear: parameter search
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
                        class_accuracy_percent, true_vs_predicted_confusion_matrix = validate_model(param_string, clf, X_validation_matrix, Y_validation_matrix)
                        accuracy_and_confusion = "_accuracy=", str(class_accuracy_percent) + "_confusion=" + str(true_vs_predicted_confusion_matrix)
                        accuracy_map[param_string] = accuracy_and_confusion # class_accuracy_percent
                    except Exception:
                        print("Unexpected error:", sys.exc_info())
                        bad = True
                        class_accuracy_percent = 0
                    accuracy_map[param_string] = class_accuracy_percent
    return


########
######## PARAMETERS
########

# svm.SVC
def C_range(isExplore):
    if(isExplore):
        return [1, 1e2, 1e3] # 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    return [1e2] # 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
def gamma_range(isExplore):
    if(isExplore):
        return [1e-8, 1e-7, 1e-6]  # 1e-5, 1e-9, 1e-11, 1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    return [1e-7]  # 1e-5, 1e-9, 1e-11, 1e-10, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9

# svm.NuSVC
def nu_range(isExplore):
    if(isExplore):
        return [1e-6, 1e-5, 1e-4, 1e-3, 0.01] # 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.20, 0.25, 0.05, 0.1, 0.15
    return [1e-6, 1e-5, 1e-4, 1e-3, 0.01] # 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.20, 0.25, 0.05, 0.1, 0.15

# svm.LinearSVC
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

# Principal component analysis
def get_pca_solver(isExplore):
    if(isExplore):
        return ['auto', 'full', 'arpack', 'randomized']
    return ['randomized']


########
######## CROSS-VALIDATION
########

# performance
def determine_model_performance(set_name, X_train, Y_train, X_validation, Y_validation,
                             param_accuracy, isExplore):
    performance_of_svm_svc(set_name, X_train, Y_train, X_validation, Y_validation,
                    C_range(isExplore), gamma_range(isExplore), param_accuracy)
    if(isExplore):
        accuracy_of_nu(set_name, X_train, Y_train, X_validation, Y_validation,
                       nu_range(isExplore), param_accuracy)
        accuracy_of_linear(set_name, X_train, Y_train, X_validation, Y_validation,
                           linear_penalty(isExplore), linear_loss(isExplore), linear_multiclass_strategy(isExplore), C_range(isExplore), param_accuracy)

# performance: against validation and training sets
def determine_performance(X_train, Y_train, X_validation, Y_validation,
                       param_accuracy, isExplore, fold_label):
    determine_model_performance(fold_label+"_trainWithTraining_validateWithValidation", X_train, Y_train, X_validation, Y_validation,
                       param_accuracy, isExplore)
    if(isExplore):
        determine_model_performance(fold_label+"_trainWithTraining_validateWithTraining", X_train, Y_train, X_train, Y_train,
                       param_accuracy, isExplore)


########
######## DATA: PROCESSING
########

# Recover from disk batch
X_full_train_set = getMatrixFromFile("counting_train", "X")
Y_full_train_set = getMatrixFromFile("counting_train", "Y")
X_full_validation_set = getMatrixFromFile("counting_val", "X")
Y_full_validation_set = getMatrixFromFile("counting_train", "Y")

# Normalize
X_train_mean_variance_normalized = getRoundedZeroMeanNormalizedVarianceMatrix(X_full_train_set)
X_validation_mean_variance_normalized = getRoundedZeroMeanNormalizedVarianceMatrix(X_full_validation_set)


# Final matrix for run
def getFinalMatrices(pca_model):
    X_train_pca_mean_variance_normalized = getPcaMatrix(X_train_mean_variance_normalized, pca_model)
    X_validation_pca_mean_variance_normalized = getPcaMatrix(X_validation_mean_variance_normalized, pca_model)
    X_train_final = X_train_pca_mean_variance_normalized
    Y_train_final = Y_full_train_set
    print (X_train_final.shape, " <= X_train_final[0]=", X_train_final[0])
    X_validation_final = X_validation_pca_mean_variance_normalized
    Y_validation_final = Y_full_validation_set
    print (X_validation_final.shape, " <= X_validation_final[0]=", X_validation_final[0])
    return X_train_final, Y_train_final

### RUN
np.set_printoptions(precision=2)
kf = KFold(n_splits=number_of_folds)

# cross-validate
def crossValidate(X_train_cross_validation, Y_train_cross_validation, cross_validation_report, pca_model_string):
    for train_index, test_index in kf.split(X_train_cross_validation):
        index_key = "PCA=",pca_model_string, "_X_fold_index_average=", str(np.average(train_index))
        # index_key = "X_fold_index=", str(train_index[0]) + "->" + str(train_index[len(train_index)-1])
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_validation = X_train_cross_validation[train_index], X_train_cross_validation[test_index]
        Y_train, Y_validation = Y_train_cross_validation[train_index], Y_train_cross_validation[test_index]
        param_accuracy = {}
        determine_performance(X_train, Y_train, X_validation, Y_validation, param_accuracy, isExplore, str(index_key))
        cross_validation_report[index_key] = param_accuracy

# SVM CLASS WEIGHT
# wclf = svm.SVC(kernel='linear', class_weight={1: 10})


# Evaluate against different principal component analysis options
param_report = {}
for pca_solver in get_pca_solver(isExplore):
    try:
        pca_model = PCA(n_components=pca_columns, svd_solver=pca_solver)
        X_train_final, Y_train_final = getFinalMatrices(pca_model)
        crossValidate(X_train_final, Y_train_final, param_report, pca_model)
    except:
        print("EVALUATE: unexpected error:", sys.exc_info())



########
######## REPORT
########

for index_key in param_report.keys():
    param_accuracy = param_report[index_key]
    for accuracy_key in param_accuracy.keys():
        print(accuracy_key, "==>", param_accuracy[accuracy_key])

print("DONE!!!")

########
# PLOT
########

# sample_name = "00001"
# sample_image = imread(images_path+sample_name+".jpg")
# plt.imshow(sample_image)
# plt.show()
