import numpy as np
# from svm_raw import get_class_weights

def get_class_weights(Y_matrix, num_classes):
    class_counts = np.zeros(num_classes)
    for y_index in range(len(Y_matrix)):
        class_id = Y_matrix[y_index]
        class_counts[class_id] = class_counts[class_id] + 1
        print("y_index=",y_index, " class_id=",class_id)
    overall_high = np.amax(class_counts)
    print("overall_high=",overall_high)
    class_weights = np.zeros(num_classes)
    for class_id in range(num_classes):
        try:
            class_weights[class_id] = int(overall_high/class_counts[class_id])
        except:
            print("overall_high=",overall_high, " class_counts[]", class_counts[class_id])
    print("class_counts=",class_counts, "class_weights=", class_weights)
    return class_weights

get_class_weights([0,1,0,0,1,0,2], 3)